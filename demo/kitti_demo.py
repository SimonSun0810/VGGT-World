"""
Minimal KITTI demo: two conditioning frames -> FM token generation -> depth.

Default inputs are the first val sequence (sorted) and the same two frames
used as conditioning in eval/kitti_val_short.py (indices 9 and 11).

Example (from repo root, vggt conda env):
  python demo/kitti_demo.py
  python demo/kitti_demo.py --frame1 /path/a.png --frame2 /path/b.png --ckpt /path/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os.path as osp
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "training_fm" / "config"
DEMO_DIR = Path(__file__).resolve().parent
OUT_DIR = DEMO_DIR / "outputs"
FRAMES_DIR = DEMO_DIR / "frames"

# Preprocessed demo frames (224x448 HxW), same resize/crop as eval/kitti_val_short.py.
DEFAULT_FRAME1 = str(FRAMES_DIR / "frame1_0000000009.png")
DEFAULT_FRAME2 = str(FRAMES_DIR / "frame2_0000000011.png")
DEFAULT_FRAME3 = str(FRAMES_DIR / "frame3_0000000013.png")
DEFAULT_FRAME4 = str(FRAMES_DIR / "frame4_0000000015.png")

DEFAULT_CKPT = ""

TARGET_H, TARGET_W = 224, 448


def load_image(path: str, target_h: int, target_w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(target_h / h, target_w / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_image_vggt(path: str, target_h: int, target_w: int, device: str) -> torch.Tensor:
    """Single-frame batch for model.forward_vggt: [1, 1, 3, H, W]."""
    return load_image(path, target_h, target_w).unsqueeze(0).unsqueeze(0).to(device)


def _apply_jet_colormap(x: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_depth_color(depth: torch.Tensor, stem: str, out_dir: Path) -> None:
    d = depth.detach().float().cpu().numpy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = np.percentile(d, 5), np.percentile(d, 95)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normed = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    color = _apply_jet_colormap(normed)
    out_path = out_dir / f"{stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color, mode="RGB").save(out_path)
    print(f"[INFO] saved {out_path}")


def load_model(ckpt_path: str, device: str, config_name: str = "default_kitti"):
    # Relative to this file (demo/); Hydra resolves against the caller script location.
    config_path = "../training_fm/config"
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg.model, _recursive_=False)
    data = torch.load(ckpt_path, map_location="cpu")
    state = data["model"] if isinstance(data, dict) and "model" in data else data
    strict = getattr(cfg.checkpoint, "strict", False)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[WARN] load missing={len(missing)} unexpected={len(unexpected)} (strict={strict})")
    print(f"[INFO] loaded checkpoint: {ckpt_path}")

    model.to(device=device)
    model.eval()
    if hasattr(model, "fm") and model.fm is not None:
        model.fm.eval()
    return model


@torch.no_grad()
def run_demo(
    model,
    frame_paths: List[str],
    device: str,
    steps: int = 50,
) -> Dict[str, torch.Tensor]:
    if len(frame_paths) != 4:
        raise ValueError("Need exactly 4 frame paths: 2 conditioning + 2 decode context.")

    sequence_frames = torch.stack(
        [load_image(p, TARGET_H, TARGET_W) for p in frame_paths], dim=0
    ).to(device)
    images = sequence_frames.unsqueeze(0)  # [B,4,3,H,W]
    img12 = images[:, :2]

    cond_stage_list, patch_start_idx = model.aggregator.part1(img12)
    tgt_stage_list, _ = model.aggregator.part1(images)
    x1_big = torch.cat(tgt_stage_list, dim=1)[:, 2:4, :, :]

    b2, t2, ntot, ctok = x1_big.shape
    dtype_fm = next(model.fm.parameters()).dtype
    shape_like = torch.zeros((b2, t2, ntot, ctok), device=device, dtype=dtype_fm)

    patch_size = model.aggregator.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size
    patch_hw = (TARGET_H // patch_h, TARGET_W // patch_w)

    gen_layers = model.fm.sample_euler(
        cond_layers=cond_stage_list,
        shape_like=shape_like,
        steps=steps,
        patch_hw=patch_hw,
    )
    gen_tokens = torch.cat(gen_layers, dim=1)
    gt_tokens = torch.cat(cond_stage_list, dim=1)[:, 0:2, :, :]
    combo_tokens = torch.cat([gt_tokens, gen_tokens], dim=1)

    agg_dtype = next(model.aggregator.parameters()).dtype
    combo_stage_list, _ = model.aggregator.part2([combo_tokens.to(agg_dtype)])

    decode_dtype = next(model.depth_head.parameters()).dtype
    combo_stage_list = [x.to(decode_dtype) for x in combo_stage_list]
    images = images.to(decode_dtype)
    depth, _ = model.depth_head(
        combo_stage_list, images=images, patch_start_idx=patch_start_idx
    )
    # Generated frames: sequence indices 2 and 3 (frame3 / frame4 inputs).
    pred_depth_frame3 = depth[0, 2, :, :, 0]
    pred_depth_frame4 = depth[0, 3, :, :, 0]

    vggt_out3 = model.forward_vggt(load_image_vggt(frame_paths[2], TARGET_H, TARGET_W, device))
    vggt_out4 = model.forward_vggt(load_image_vggt(frame_paths[3], TARGET_H, TARGET_W, device))
    vggt_depth_frame3 = vggt_out3["depth"][0, 0, :, :, 0]
    vggt_depth_frame4 = vggt_out4["depth"][0, 0, :, :, 0]

    return {
        "pred_depth_frame3": pred_depth_frame3,
        "pred_depth_frame4": pred_depth_frame4,
        "vggt_depth_frame3": vggt_depth_frame3,
        "vggt_depth_frame4": vggt_depth_frame4,
    }


def main():
    parser = argparse.ArgumentParser(description="KITTI two-frame FM depth demo.")
    parser.add_argument("--frame1", default=DEFAULT_FRAME1, help="First conditioning frame.")
    parser.add_argument("--frame2", default=DEFAULT_FRAME2, help="Second conditioning frame.")
    parser.add_argument("--frame3", default=DEFAULT_FRAME3, help="Context frame for depth decode.")
    parser.add_argument("--frame4", default=DEFAULT_FRAME4, help="Context frame for depth decode.")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="KITTI checkpoint (.pt).")
    parser.add_argument("--config", default="default_kitti", help="Hydra config name.")
    parser.add_argument("--steps", type=int, default=50, help="FM Euler steps.")
    parser.add_argument("--out_dir", default=str(OUT_DIR), help="Output directory.")
    args = parser.parse_args()

    frame_paths = [args.frame1, args.frame2, args.frame3, args.frame4]
    for p in frame_paths:
        if not osp.isfile(p):
            raise FileNotFoundError(f"Frame not found: {p}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print("[INFO] conditioning frames:")
    print(f"  frame1: {args.frame1}")
    print(f"  frame2: {args.frame2}")

    model = load_model(args.ckpt, device, config_name=args.config)
    outputs = run_demo(model, frame_paths, device, steps=args.steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, depth in outputs.items():
        save_depth_color(depth, stem, out_dir)


if __name__ == "__main__":
    main()
