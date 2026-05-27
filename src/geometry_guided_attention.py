"""Geometry-Guided Cross-View Attention (GCA)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GeometryGuidedCrossViewAttention(nn.Module):
    """
    Implements Geometry-Guided Cross-View Attention from the paper.

    A geometry-indexed proxy feature is sampled from the reference view and
    used as the query. Keys and values are sampled from a local window centered
    at the same correspondence location.
    """
    def __init__(
        self,
        hidden_dim,
        num_heads=8,
        neighborhood_size=5,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.neighborhood_size = neighborhood_size
        self.k = neighborhood_size // 2

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.rel_pos_embed = nn.Parameter(
            torch.randn(1, num_heads, 1, neighborhood_size * neighborhood_size, self.head_dim) * 0.02
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def init_from_unet_attn(self, unet_attn1, unet_norm1=None):
        """Initialize GCA projections from a pretrained UNet self-attention block."""
        with torch.no_grad():
            self.q_proj.weight.copy_(unet_attn1.to_q.weight)
            if unet_attn1.to_q.bias is not None:
                self.q_proj.bias.copy_(unet_attn1.to_q.bias)
            else:
                self.q_proj.bias.zero_()

            self.k_proj.weight.copy_(unet_attn1.to_k.weight)
            if unet_attn1.to_k.bias is not None:
                self.k_proj.bias.copy_(unet_attn1.to_k.bias)
            else:
                self.k_proj.bias.zero_()

            self.v_proj.weight.copy_(unet_attn1.to_v.weight)
            if unet_attn1.to_v.bias is not None:
                self.v_proj.bias.copy_(unet_attn1.to_v.bias)
            else:
                self.v_proj.bias.zero_()

            self.out_proj.weight.copy_(unet_attn1.to_out[0].weight)
            self.out_proj.bias.copy_(unet_attn1.to_out[0].bias)

            if unet_norm1 is not None:
                self.norm.weight.copy_(unet_norm1.weight)
                self.norm.bias.copy_(unet_norm1.bias)

        print(f"[GCA] Initialized from UNet self-attention: dim={self.hidden_dim}, heads={self.num_heads}")

    def sample_proxy_features(self, reference_feature_map, correspondence, validity_mask, H, W):
        """Sample geometry-indexed proxy features from the reference map."""
        B, C, _, _ = reference_feature_map.shape

        with torch.cuda.amp.autocast(enabled=False):
            grid = correspondence.clone().float()
            grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
            grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0

            valid_x = (grid[:, 0:1] >= -1) & (grid[:, 0:1] <= 1)
            valid_y = (grid[:, 1:2] >= -1) & (grid[:, 1:2] <= 1)
            valid_coords = (valid_x & valid_y).float()
            valid_proxy_mask = valid_coords * validity_mask

            grid = rearrange(grid, 'b c h w -> b h w c')

        proxy_features = F.grid_sample(
            reference_feature_map.float(), grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).to(reference_feature_map.dtype)

        return proxy_features, valid_proxy_mask

    def sample_local_reference_window(self, reference_feature_map, correspondence, mask, H, W):
        """Sample projected keys and values from the local GCA window."""
        B, C, _, _ = reference_feature_map.shape
        K = self.neighborhood_size
        k = self.k
        device = reference_feature_map.device
        dtype = reference_feature_map.dtype

        reference_tokens = rearrange(reference_feature_map, 'b c h w -> b (h w) c')
        reference_tokens_normed = self.norm(reference_tokens)
        reference_keys = self.k_proj(reference_tokens_normed)
        reference_values = self.v_proj(reference_tokens_normed)

        reference_keys_2d = rearrange(reference_keys, 'b (h w) c -> b c h w', h=H, w=W)
        reference_values_2d = rearrange(reference_values, 'b (h w) c -> b c h w', h=H, w=W)

        with torch.cuda.amp.autocast(enabled=False):
            offsets = torch.stack(torch.meshgrid(
                torch.arange(-k, k+1, device=device, dtype=torch.float32),
                torch.arange(-k, k+1, device=device, dtype=torch.float32),
                indexing='xy'
            ), dim=-1).reshape(-1, 2)

            base_coords = rearrange(correspondence.float(), 'b c h w -> b h w c')
            neighbor_coords = base_coords.unsqueeze(-2) + offsets.reshape(1, 1, 1, -1, 2)

            neighbor_coords_norm = neighbor_coords.clone()
            neighbor_coords_norm[..., 0] = 2.0 * neighbor_coords_norm[..., 0] / (W - 1) - 1.0
            neighbor_coords_norm[..., 1] = 2.0 * neighbor_coords_norm[..., 1] / (H - 1) - 1.0

            valid = (
                (neighbor_coords[..., 0] >= 0) & (neighbor_coords[..., 0] < W) &
                (neighbor_coords[..., 1] >= 0) & (neighbor_coords[..., 1] < H)
            )

            neighbor_coords_flat = rearrange(neighbor_coords_norm, 'b h w k c -> b (h w k) 1 c')

            local_keys = F.grid_sample(
                reference_keys_2d.float(), neighbor_coords_flat,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )
            local_values = F.grid_sample(
                reference_values_2d.float(), neighbor_coords_flat,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )

        local_keys = rearrange(local_keys, 'b c (n k) 1 -> b n k c', n=H*W, k=K*K).to(dtype)
        local_values = rearrange(local_values, 'b c (n k) 1 -> b n k c', n=H*W, k=K*K).to(dtype)

        valid_mask = rearrange(valid, 'b h w k -> b (h w) k')
        base_mask = rearrange(mask, 'b 1 h w -> b (h w) 1')
        valid_mask = valid_mask & (base_mask > 0.5)

        return local_keys, local_values, valid_mask

    def forward(self, target_tokens, reference_tokens, correspondence, validity_mask, spatial_size):
        B, N, C = target_tokens.shape
        H, W = spatial_size

        reference_feature_map = rearrange(reference_tokens, 'b (h w) c -> b c h w', h=H, w=W)

        proxy_features, valid_proxy_mask = self.sample_proxy_features(reference_feature_map, correspondence, validity_mask, H, W)
        proxy_features_tokens = rearrange(proxy_features, 'b c h w -> b (h w) c')
        valid_proxy_mask_flat = rearrange(valid_proxy_mask, 'b 1 h w -> b (h w) 1')

        Q = self.q_proj(self.norm(proxy_features_tokens))
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)

        local_keys, local_values, valid_mask = self.sample_local_reference_window(
            reference_feature_map, correspondence, validity_mask, H, W
        )

        K = rearrange(local_keys, 'b n k (h d) -> b h n k d', h=self.num_heads)
        V = rearrange(local_values, 'b n k (h d) -> b h n k d', h=self.num_heads)
        K = K + self.rel_pos_embed

        Q_expanded = Q.unsqueeze(-2)
        attn_scores = torch.matmul(Q_expanded, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.squeeze(-2)

        valid_mask_expanded = valid_mask.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(~valid_mask_expanded, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        attn_weights_expanded = attn_weights.unsqueeze(-1)
        out = (attn_weights_expanded * V).sum(dim=-2)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)

        out = out * valid_proxy_mask_flat

        return out
