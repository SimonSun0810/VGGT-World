"""Validation-time feature / depth / point-cloud dump helpers."""

import os

import cv2
import numpy as np
import torch
from PIL import Image


def _minmax_uint8(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def _to_B1NC(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize to [B,1,N,C].
    Accepts:
      - [B,1,N,C]
      - [B,T,N,C]  -> take T=0
      - [1,N,C] or [N,C] -> add batch/time
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim == 4:
        B, T, N, C = x.shape
        if T == 1:
            return x
        return x[:, :1, :, :]
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")


def _tokens_to_grid(x_b1nc: torch.Tensor, grid_hw=37, extra_tokens: int = 5, drop: str = "tail"):
    """
    x_b1nc: [B,1,N,C] where N = grid_hw*grid_hw + extra_tokens
    returns: grid [B,1,grid_hw,grid_hw,C]
    """
    assert x_b1nc.ndim == 4, x_b1nc.shape
    B, T, N, C = x_b1nc.shape
    assert T == 1, f"Need T=1, got {T}"
    if isinstance(grid_hw, (tuple, list)):
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    else:
        grid_h = grid_w = int(grid_hw)
    N_patch = grid_h * grid_w
    assert N == N_patch + extra_tokens, f"Expected N={N_patch}+{extra_tokens}={N_patch+extra_tokens}, got {N}"

    if drop == "tail":
        patch = x_b1nc[:, :, :N_patch, :]
    elif drop == "head":
        patch = x_b1nc[:, :, extra_tokens:, :]
    else:
        raise ValueError('drop must be "tail" or "head"')

    grid = patch.view(B, 1, grid_h, grid_w, C)
    return grid


def _dump_grid(grid: torch.Tensor, out_dir: str, name: str, save_npy: bool = True, max_B: int = 4):
    """
    grid: [B,1,H,W,C]
    Saves:
      - {name}.bXXX.norm.png
      - {name}.bXXX.pca.png
    """
    os.makedirs(out_dir, exist_ok=True)
    assert grid.ndim == 5, grid.shape
    B, T, H, W, C = grid.shape
    assert T == 1, T

    b_lim = min(B, max_B)

    for b in range(b_lim):
        g = grid[b, 0].detach().float().cpu()
        x = g.view(-1, C)

        norm = torch.linalg.norm(g, dim=-1).numpy()
        Image.fromarray(_minmax_uint8(norm), mode="L").save(
            os.path.join(out_dir, f"{name}.b{b:03d}.norm.png")
        )

        x0 = x - x.mean(dim=0, keepdim=True)
        U, S, V = torch.pca_lowrank(x0, q=3, center=False)
        rgb = (x0 @ V[:, :3]).view(H, W, 3).numpy()

        w0 = _minmax_uint8(rgb[..., 0]).astype(np.float32) / 255.0
        w1 = _minmax_uint8(rgb[..., 1]).astype(np.float32) / 255.0
        w2 = _minmax_uint8(rgb[..., 2]).astype(np.float32) / 255.0
        palette = np.array(
            [
                [255.0, 0.0, 0.0],
                [0.0, 0.0, 255.0],
                [255.0, 255.0, 0.0],
            ],
            dtype=np.float32,
        )
        rgb_mix = (
            w0[..., None] * palette[0]
            + w1[..., None] * palette[1]
            + w2[..., None] * palette[2]
        )
        rgb_u8 = np.clip(rgb_mix, 0, 255).astype(np.uint8)
        Image.fromarray(rgb_u8, mode="RGB").save(
            os.path.join(out_dir, f"{name}.b{b:03d}.pca.png")
        )


def save_feature(
    feature1: torch.Tensor,
    feature2: torch.Tensor,
    out_dir: str,
    prefix1: str = "x1_big",
    prefix2: str = "gen",
    grid_hw=37,
    extra_tokens: int = 5,
    drop: str = "head",
    max_B: int = 4,
):
    """
    Dump token features as norm/PCA grid images.

    Assumes both are [B,1,(H*W+extra_tokens),1024] or [B,T,...] (T>=1).
    """
    os.makedirs(out_dir, exist_ok=True)

    t1 = feature1.shape[1] if feature1.ndim == 4 else 1
    t2 = feature2.shape[1] if feature2.ndim == 4 else 1
    t_len = min(t1, t2)

    last_g1 = None
    last_g2 = None
    for t in range(t_len):
        f1_in = feature1[:, t:t + 1] if feature1.ndim == 4 else feature1
        f2_in = feature2[:, t:t + 1] if feature2.ndim == 4 else feature2

        f1 = _to_B1NC(f1_in)
        f2 = _to_B1NC(f2_in)

        assert f1.shape[-1] == 1024, f"feature1 C != 1024: {f1.shape}"
        assert f2.shape[-1] == 1024, f"feature2 C != 1024: {f2.shape}"
        if isinstance(grid_hw, (tuple, list)):
            grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        else:
            grid_h = grid_w = int(grid_hw)
        expected_n = grid_h * grid_w + extra_tokens
        assert f1.shape[2] == expected_n, f"feature1 N mismatch: {f1.shape}"
        assert f2.shape[2] == expected_n, f"feature2 N mismatch: {f2.shape}"

        g1 = _tokens_to_grid(f1, grid_hw=grid_hw, extra_tokens=extra_tokens, drop=drop)
        g2 = _tokens_to_grid(f2, grid_hw=grid_hw, extra_tokens=extra_tokens, drop=drop)

        suffix = f"_t{t}" if t_len > 1 else ""
        _dump_grid(g1, out_dir, f"{prefix1}{suffix}", save_npy=True, max_B=max_B)
        _dump_grid(g2, out_dir, f"{prefix2}{suffix}", save_npy=True, max_B=max_B)

        last_g1, last_g2 = g1, g2

    return last_g1, last_g2


def save_depth_png(depth: np.ndarray, out_png: str):
    """depth: [H,W] float, normalized for visualization."""
    d = depth.copy()
    d[np.isnan(d)] = 0
    d[np.isinf(d)] = 0
    vmin = np.percentile(d, 5)
    vmax = np.percentile(d, 95)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    d = (d - vmin) / (vmax - vmin)
    d = np.clip(d, 0, 1)
    img = (d * 255).astype(np.uint8)
    cv2.imwrite(out_png, img)


def write_ply_xyz(points: np.ndarray, out_ply: str):
    """points: [N,3] float"""
    points = points.astype(np.float32)
    with open(out_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
