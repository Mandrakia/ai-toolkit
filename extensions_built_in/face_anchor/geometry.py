"""Differentiable face alignment geometry.

Validated against insightface in smoke tests (2026-05-21):
  - estimate_norm reproduces insightface.utils.face_align (same arcface_dst template)
  - warp_batch (grid_sample) reproduces insightface norm_crop to cos 0.9998+, profile included
  - fixed-window crop in latent + kps mapped into tile space preserves cos (tile==full-frame)
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import SimilarityTransform

# ArcFace 5-point reference template for a 112x112 aligned crop (insightface arcface_dst)
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def estimate_norm(kps_px: np.ndarray) -> np.ndarray:
    """5 landmark points (in tile-pixel coords) -> 2x3 similarity transform (src->dst=112 template)."""
    t = SimilarityTransform()
    t.estimate(kps_px.astype(np.float32), ARCFACE_DST)
    return t.params[:2].astype(np.float32)


def fixed_window_latent(center_xy_latent, B, lat_h, lat_w):
    """Top-left (lx0, ly0) of a BxB latent window centered on the face, clamped into the grid."""
    cx, cy = center_xy_latent
    lx0 = int(np.clip(round(cx - B / 2), 0, max(0, lat_w - B)))
    ly0 = int(np.clip(round(cy - B / 2), 0, max(0, lat_h - B)))
    return lx0, ly0


def warp_batch(tiles: torch.Tensor, Ms: torch.Tensor, out: int = 112, dst: torch.Tensor = None) -> torch.Tensor:
    """Batched differentiable warp. tiles (n,3,H,W) float; Ms (n,2,3) src->dst. -> (n,3,out,out).

    `dst` is the (3, out*out) homogeneous destination grid; pass a precomputed one to avoid
    rebuilding the fixed meshgrid every call.
    """
    n, _, H, W = tiles.shape
    dev = tiles.device
    M3 = torch.eye(3, device=dev).unsqueeze(0).repeat(n, 1, 1)
    M3[:, :2, :] = Ms.to(dev).float()
    Minv = torch.linalg.inv(M3)[:, :2, :]                      # (n,2,3) dst->src
    if dst is None:
        ys, xs = torch.meshgrid(torch.arange(out, device=dev, dtype=torch.float32),
                                torch.arange(out, device=dev, dtype=torch.float32), indexing="ij")
        dst = torch.stack([xs.flatten(), ys.flatten(), torch.ones(out * out, device=dev)], 0)  # (3, out*out)
    src = Minv @ dst                                           # (n,2,out*out)
    gx = (src[:, 0] + 0.5) / W * 2 - 1
    gy = (src[:, 1] + 0.5) / H * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(n, out, out, 2)
    return F.grid_sample(tiles.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
