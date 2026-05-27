import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from geometry_guided_attention import GeometryGuidedCrossViewAttention


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, neighborhood_size=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.geo_attn = GeometryGuidedCrossViewAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            neighborhood_size=neighborhood_size
        )
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        nn.init.zeros_(self.weight_predictor[-1].weight)
        nn.init.constant_(self.weight_predictor[-1].bias, -2.5)

    def init_from_unet_attn(self, unet_attn1, unet_norm1=None):
        """Initialize GeoQuery attention from the UNet self-attention block."""
        self.geo_attn.init_from_unet_attn(unet_attn1)

        if unet_norm1 is not None:
            with torch.no_grad():
                self.geo_attn.norm.weight.copy_(unet_norm1.weight)
                self.geo_attn.norm.bias.copy_(unet_norm1.bias)
                print("[GeoQuery] Initialized geo_attn.norm from UNet norm1")

        print(f"[GeoQuery] Initialized from UNet: dim={self.hidden_dim}, heads={self.num_heads}")

    def forward(self, target_tokens, reference_tokens, self_attention_out, correspondence, validity_mask, spatial_size):
        """
        Args:
            target_tokens: target-view tokens.
            reference_tokens: reference-view tokens.
            self_attention_out: target tokens after multi-view self-attention.
            correspondence: geometric correspondence field from target to reference.
            validity_mask: correspondence validity mask.
            spatial_size: token-grid height and width.

        Returns:
            output: [B, N, D]
        """
        gca_features = self.geo_attn(target_tokens, reference_tokens, correspondence, validity_mask, spatial_size)
        self_attention_delta = self_attention_out - target_tokens

        weight = torch.sigmoid(self.weight_predictor(
            torch.cat([gca_features, self_attention_delta], dim=-1)
        ))

        valid_correspondence_flat = rearrange(validity_mask, 'b 1 h w -> b (h w) 1')
        weight = weight * valid_correspondence_flat

        output = target_tokens + (1-weight) * self_attention_delta + weight * gca_features
        return output
