import argparse
import glob
import os.path as osp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "training_fm" / "config"
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image
from vggt.models.vggt import VGGT
from typing import Optional
from hydra import compose, initialize
from hydra.utils import instantiate


def build_sequence_list(kitti_root: str, split: str):
    if split == "test":
        split = "val"

    split_dir = osp.join(kitti_root, split)
    image_02_dirs = glob.glob(
        osp.join(split_dir, "**", "image_02", "data"),
        recursive=True,
    )
    sequence_list = sorted(
        osp.relpath(image_dir, split_dir)
        for image_dir in image_02_dirs
        if osp.isdir(image_dir)
    )
    return sequence_list


def main():
    parser = argparse.ArgumentParser(description="Build KITTI val sequence list.")
    parser.add_argument(
        "--kitti_root",
        default="",
        help="Root of KITTI raw data (parent of val/).",
    )
    parser.add_argument("--split", default="val", help="Dataset split: train/val/test")
    parser.add_argument(
        "--ckpt",
        default="",
    )
    args = parser.parse_args()

    sequence_list = build_sequence_list(args.kitti_root, args.split)

    split = "val" if args.split == "test" else args.split
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model_from_config(config_name: str, ckpt_path: Optional[str], device: str):
        with initialize(version_base=None, config_path=str(CONFIG_DIR)):
            cfg = compose(config_name=config_name)

        model = instantiate(cfg.model, _recursive_=False)

        full_ckpt = ckpt_path
        if not full_ckpt:
            raise ValueError("ckpt_path is empty and cfg.checkpoint.resume_checkpoint_path is not set.")

        data = torch.load(full_ckpt, map_location="cpu")
        state = data["model"] if isinstance(data, dict) and "model" in data else data

        strict = getattr(cfg.checkpoint, "strict", False)
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print(f"[WARN] load missing={len(missing)} unexpected={len(unexpected)} (strict={strict})")
        print(f"[INFO] initialized full model from {full_ckpt} [online]")

        model.to(device=device)
        model.eval()
        if hasattr(model, "fm") and model.fm is not None:
            model.fm.eval()
        return model, cfg

    model, cfg = load_model_from_config("default", args.ckpt, device)
    abs_rels_gt = []
    delta1s_gt = []
    abs_rels_vggt = []
    delta1s_vggt = []
    abs_rels_vggt_gt = []
    delta1s_vggt_gt = []

    for idx, sequence_name in enumerate(sequence_list):
        splits = sequence_name.split("_")
        frames_dir = osp.join(args.kitti_root, split, sequence_name)
        frames_filepaths = sorted(glob.glob(osp.join(frames_dir, "*.png")))
        if not frames_filepaths:
            frames_filepaths = sorted(
                glob.glob(osp.join(frames_dir, "*.jpg"))
                + glob.glob(osp.join(frames_dir, "*.jpeg"))
                + glob.glob(osp.join(frames_dir, "*.PNG"))
                + glob.glob(osp.join(frames_dir, "*.JPG"))
                + glob.glob(osp.join(frames_dir, "*.JPEG"))
            )


        # Select frames: idx 5 7 9 11 13 
        start_idx = 5  # 0-based index for the 14th frame
        step = 2
        sequence_length = 5
        if len(frames_filepaths) < start_idx + step * sequence_length:
            raise ValueError(
                f"Not enough frames for {sequence_name}: have {len(frames_filepaths)}"
            )
        sequence_frames_path = frames_filepaths[
            start_idx : start_idx + step * sequence_length : step
        ]

        # Load, resize by short side, center-crop long side to 224x448 (H,W)
        target_h, target_w = 224, 448
        sequence_frames = []
        for f in sequence_frames_path:
            img = Image.open(f).convert("RGB")
            w, h = img.size
            scale = max(target_h / h, target_w / w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = img.resize((new_w, new_h), Image.BICUBIC)

            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))
            sequence_frames.append(np.array(img, dtype=np.float32) / 255.0)

        sequence_frames = np.stack(sequence_frames, axis=0)  # [T,H,W,3]
        sequence_frames = (
            torch.from_numpy(sequence_frames).permute(0, 3, 1, 2).contiguous()
        ).to(device)

        # Use frame 13 (index 13) as GT depth source for evaluation
        gt_path = frames_filepaths[13]
        gt_img = Image.open(gt_path).convert("RGB")
        w, h = gt_img.size
        scale = max(target_h / h, target_w / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        gt_img = gt_img.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        gt_img = gt_img.crop((left, top, left + target_w, top + target_h))
        gt_img = np.array(gt_img, dtype=np.float32) / 255.0
        gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)

        # FM stage 1: use frames 5 & 7 to predict 9 & 11
        images_stage1 = sequence_frames[0:4].unsqueeze(0)  # [B,4,3,H,W] -> 5,7,9,11
        img12 = images_stage1[:, :2]

        cond_stage_list_5_7, patch_start_idx = model.aggregator.part1(img12)
        tgt_stage_list_stage1, _ = model.aggregator.part1(images_stage1)
        x1_big = torch.cat(tgt_stage_list_stage1, dim=1)  # [B,4,Ntot,C]
        x1_big = x1_big[:, 2:4, :, :]                     # [B,2,Ntot,C]

        B2, T2, Ntot, Ctok = x1_big.shape
        dtype_fm = next(model.fm.parameters()).dtype
        shape_like = torch.zeros((B2, T2, Ntot, Ctok), device=device, dtype=dtype_fm)

        patch_size = model.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        patch_hw = (target_h // patch_h, target_w // patch_w)

        gen_layers_stage1 = model.fm.sample_euler(
            cond_layers=cond_stage_list_5_7,
            shape_like=shape_like,
            steps=50,
            patch_hw=patch_hw,
        )
        gen_tokens_stage1 = torch.cat(gen_layers_stage1, dim=1)  # [B,2,Ntot,C] -> 9,11

        # FM stage 2: use frame 7 (GT) + frame 9 (pred) to predict 11 & 13
        cond_stage_list_7_9 = []
        for cond_layer, gen_layer in zip(cond_stage_list_5_7, gen_layers_stage1):
            cond_7 = cond_layer[:, 1:2, :, :]
            pred_9 = gen_layer[:, 0:1, :, :]
            cond_stage_list_7_9.append(torch.cat([cond_7, pred_9], dim=1))

        images_stage2 = sequence_frames[1:5].unsqueeze(0)  # [B,4,3,H,W] -> 7,9,11,13
        tgt_stage_list_stage2, _ = model.aggregator.part1(images_stage2)
        x2_big = torch.cat(tgt_stage_list_stage2, dim=1)  # [B,4,Ntot,C]
        x2_big = x2_big[:, 2:4, :, :]                     # [B,2,Ntot,C]

        B2, T2, Ntot, Ctok = x2_big.shape
        shape_like2 = torch.zeros((B2, T2, Ntot, Ctok), device=device, dtype=dtype_fm)

        gen_layers_stage2 = model.fm.sample_euler(
            cond_layers=cond_stage_list_7_9,
            shape_like=shape_like2,
            steps=50,
            patch_hw=patch_hw,
        )
        gen_tokens_stage2 = torch.cat(gen_layers_stage2, dim=1)  # [B,2,Ntot,C] -> 11,13

        gen_11_from_stage1 = gen_tokens_stage1[:, 1:2, :, :]
        gen_13_from_stage2 = gen_tokens_stage2[:, 1:2, :, :]
        combo_tokens = torch.cat(
            [
                # cond_stage_list_5_7[0][:, 0:1, :, :],
                cond_stage_list_7_9[0][:, 0:1, :, :],
                cond_stage_list_7_9[0][:, 1:2, :, :],
                gen_11_from_stage1,
                gen_13_from_stage2,
            ],
            dim=1,
        )  # [B,4,Ntot,C]

        agg_dtype = next(model.aggregator.parameters()).dtype
        combo_stage_list, _ = model.aggregator.part2([combo_tokens.to(agg_dtype)])

        decode_dtype = next(model.depth_head.parameters()).dtype
        combo_stage_list = [x.to(decode_dtype) for x in combo_stage_list]
        images_decode = sequence_frames[1:5].unsqueeze(0).to(decode_dtype)
        depth, _ = model.depth_head(
            combo_stage_list, images=images_decode, patch_start_idx=patch_start_idx
        )
        pred_depth = depth[0, 3, :, :, 0].detach().clone()  # frame 13 (0-based)

        # VGGT depth prediction on the selected frame
        gt_img = gt_img.to(device)
        vggt_out = model.forward_vggt(gt_img)
        vggt_depth = vggt_out["depth"][0, 0, :, :, 0].detach().clone()

        # Load KITTI projected depth (val_depth) for the same frame
        frame_name = osp.splitext(osp.basename(gt_path))[0]
        drive_dir = osp.dirname(osp.dirname(frames_dir))
        date_dir = osp.dirname(drive_dir)
        date_name = osp.basename(date_dir)
        drive_name = osp.basename(drive_dir)
        depth_path = osp.join(
            args.kitti_root,
            "val_depth",
            date_name,
            drive_name,
            "proj_depth",
            "groundtruth",
            "image_02",
            frame_name + ".png",
        )
        if not osp.exists(depth_path):
            print(f"[WARN] missing depth: {depth_path}")
            continue

        depth_raw = Image.open(depth_path)
        depth_png = np.array(depth_raw, dtype=np.uint16)
        depth_raw = depth_png.astype(np.float32) / 256.0
        depth_raw[depth_png == 0] = -1.0
        depth_raw = Image.fromarray(depth_raw)
        depth_raw = depth_raw.resize((new_w, new_h), Image.NEAREST)
        depth_raw = depth_raw.crop((left, top, left + target_w, top + target_h))
        depth_raw = np.array(depth_raw, dtype=np.float32)
        gt_depth = torch.from_numpy(depth_raw).to(device)

        eps = 0
        valid_gt = gt_depth > eps
        valid_vggt = vggt_depth > eps

        pred_depth_raw = pred_depth
        vggt_depth_raw = vggt_depth

        pred_depth_scaled = pred_depth_raw
        vggt_depth_scaled = vggt_depth_raw
        
        if valid_gt.any():
            pred_scale = torch.median(gt_depth[valid_gt]) / torch.median(
                pred_depth_raw[valid_gt]
            )
            vggt_scale = torch.median(gt_depth[valid_gt]) / torch.median(
                vggt_depth_raw[valid_gt]
            )
            pred_depth_scaled = pred_depth_raw * pred_scale
            vggt_depth_scaled = vggt_depth_raw * vggt_scale

        abs_rel_gt = torch.mean(
            torch.abs(pred_depth_scaled[valid_gt] - gt_depth[valid_gt]) / gt_depth[valid_gt]
        ).item()
        ratio_gt = torch.maximum(
            pred_depth_scaled / gt_depth, gt_depth / pred_depth_scaled
        )
        delta1_gt = torch.mean((ratio_gt < 1.25)[valid_gt].float()).item()

        abs_rel_vggt = torch.mean(
            torch.abs(pred_depth_raw[valid_vggt] - vggt_depth_raw[valid_vggt])
            / vggt_depth_raw[valid_vggt]
        ).item()
        ratio_vggt = torch.maximum(
            pred_depth_raw / vggt_depth_raw, vggt_depth_raw / pred_depth_raw
        )
        delta1_vggt = torch.mean((ratio_vggt < 1.25)[valid_vggt].float()).item()
        valid_vggt_gt = valid_gt & valid_vggt
        abs_rel_vggt_gt = torch.mean(
            torch.abs(vggt_depth_scaled[valid_vggt_gt] - gt_depth[valid_vggt_gt])
            / gt_depth[valid_vggt_gt]
        ).item()
        ratio_vggt_gt = torch.maximum(
            vggt_depth_scaled / gt_depth, gt_depth / vggt_depth_scaled
        )
        delta1_vggt_gt = torch.mean((ratio_vggt_gt < 1.25)[valid_vggt_gt].float()).item()

        abs_rels_gt.append(abs_rel_gt)
        delta1s_gt.append(delta1_gt)
        abs_rels_vggt.append(abs_rel_vggt)
        delta1s_vggt.append(delta1_vggt)
        abs_rels_vggt_gt.append(abs_rel_vggt_gt)
        delta1s_vggt_gt.append(delta1_vggt_gt)
        print(f"idx={idx} seq={sequence_name}")
        print(f"{'pair':<12} {'abs_rel':>10} {'delta1':>10}")
        print(
            f"{'pred-gt':<12} {abs_rel_gt:>10.6f} {delta1_gt:>10.6f}"
        )
        print(
            f"{'pred-vggt':<12} {abs_rel_vggt:>10.6f} {delta1_vggt:>10.6f}"
        )
        print(
            f"{'vggt-gt':<12} {abs_rel_vggt_gt:>10.6f} {delta1_vggt_gt:>10.6f}"
        )


    if abs_rels_gt:
        mean_abs_rel_gt = float(np.mean(abs_rels_gt))
        mean_delta1_gt = float(np.mean(delta1s_gt))
        mean_abs_rel_vggt = float(np.mean(abs_rels_vggt))
        mean_delta1_vggt = float(np.mean(delta1s_vggt))
        mean_abs_rel_vggt_gt = float(np.mean(abs_rels_vggt_gt))
        mean_delta1_vggt_gt = float(np.mean(delta1s_vggt_gt))
        print("MEAN")
        print(f"{'pair':<12} {'abs_rel':>10} {'delta1':>10}")
        print(
            f"{'pred-gt':<12} {mean_abs_rel_gt:>10.6f} {mean_delta1_gt:>10.6f}"
        )
        print(
            f"{'pred-vggt':<12} {mean_abs_rel_vggt:>10.6f} {mean_delta1_vggt:>10.6f}"
        )
        print(
            f"{'vggt-gt':<12} {mean_abs_rel_vggt_gt:>10.6f} {mean_delta1_vggt_gt:>10.6f}"
        )





if __name__ == "__main__":
    main()
