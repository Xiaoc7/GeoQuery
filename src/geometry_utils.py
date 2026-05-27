import numpy as np
import torch

from colmap_loader import qvec2rotmat
from softsplat import softsplat


def colmap_params_to_intrinsics(params: np.ndarray):
    return torch.tensor(
        [
            [params[0], 0, params[2]],
            [0, params[1], params[3]],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def colmap_pose_to_c2w(qvec, tvec):
    R_w2c = qvec2rotmat(qvec)
    t_w2c = np.array(tvec).reshape(3, 1)

    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3:] = t_c2w
    return c2w


def resize_intrinsics(K, original_size, target_size, crop_size=None):
    W_orig, H_orig = original_size
    W_target, H_target = target_size

    scale_x = W_target / W_orig
    scale_y = H_target / H_orig

    K_new = K.copy() if isinstance(K, np.ndarray) else K.clone()
    is_batch = len(K_new.shape) == 3

    if is_batch:
        K_new[:, 0, 0] *= scale_x
        K_new[:, 1, 1] *= scale_y
        K_new[:, 0, 2] *= scale_x
        K_new[:, 1, 2] *= scale_y

        if crop_size is not None:
            W_crop, H_crop = crop_size
            K_new[:, 0, 2] -= (W_target - W_crop) // 2
            K_new[:, 1, 2] -= (H_target - H_crop) // 2
    else:
        K_new[0, 0] *= scale_x
        K_new[1, 1] *= scale_y
        K_new[0, 2] *= scale_x
        K_new[1, 2] *= scale_y

        if crop_size is not None:
            W_crop, H_crop = crop_size
            K_new[0, 2] -= (W_target - W_crop) // 2
            K_new[1, 2] -= (H_target - H_crop) // 2

    return K_new


def backproject_depth(depth, K):
    B, _, H, W = depth.shape
    device = depth.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    pix_coords = torch.stack([x, y, ones], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

    K_inv = torch.inverse(K)
    cam_coords = K_inv @ pix_coords.view(B, 3, -1)
    cam_coords = cam_coords.view(B, 3, H, W)

    return cam_coords * depth


def build_geometric_correspondence(ref_coord_map, ref_depth, K, ref_pose, target_pose, alpha=0.5):
    B, _, H, W = ref_coord_map.shape
    device = ref_coord_map.device

    points_3d = backproject_depth(ref_depth, K).view(B, 3, -1)
    ones = torch.ones_like(points_3d[:, :1])
    homo_points = torch.cat([points_3d, ones], dim=1)

    world_coords = ref_pose @ homo_points
    target_cam = torch.inverse(target_pose) @ world_coords
    target_cam = target_cam[:, :3]

    invalid = (target_cam[:, 2:3] <= 0) | (ref_depth.view(B, 1, -1) <= 0)

    projected = K @ target_cam
    projected = projected[:, :2] / projected[:, 2:3].clamp(min=1e-8)
    projected[invalid.expand(-1, 2, -1)] = -1000000.0

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    x = x.to(device).float().view(1, -1).expand(B, -1)
    y = y.to(device).float().view(1, -1).expand(B, -1)

    flow = torch.zeros(B, 2, H, W, device=device)
    flow[:, 0] = (projected[:, 0] - x).view(B, H, W)
    flow[:, 1] = (projected[:, 1] - y).view(B, H, W)

    depth_z = target_cam[:, 2:3]
    importance = alpha / depth_z.clamp(min=1e-6)
    importance[invalid] = 0.0
    importance = importance - importance.amin(dim=2, keepdim=True)
    importance = importance / (importance.amax(dim=2, keepdim=True) + 1e-6)
    importance = (importance * 10 - 10).view(B, 1, H, W)

    correspondence = softsplat(ref_coord_map, flow, importance, "soft")
    invalid_mask = (correspondence == 0.0).all(dim=1, keepdim=True).to(ref_coord_map.dtype)

    return {
        "correspondence": correspondence,
        "validity_mask": 1.0 - invalid_mask,
    }
