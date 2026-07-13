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


def _prepare_frame(path: str, target_h: int, target_w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(target_h / h, target_w / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))
    img = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().unsqueeze(0).unsqueeze(0)


def build_sequence_list(cityscapes_dir: str, split: str):
    """
    Build evaluation sequence IDs from Cityscapes leftImg8bit_sequence.
    Filenames follow ``<city>_<seq>_<frame>_leftImg8bit.png``. For cities with
    many sequences (>=10), we keep one entry per sequence. For smaller cities,
    we subsample start-frame IDs every 30 frames within each sequence to avoid
    an overly sparse evaluation set.
    """
    sequences = set()
    for city_folder in glob.glob(osp.join(cityscapes_dir, split, "*")):
        city_name = osp.basename(city_folder)
        frames_in_city = glob.glob(osp.join(city_folder, "*"))
        # e.g. "frankfurt_000001" from "frankfurt_000001_000019_leftImg8bit.png"
        city_seqs = set(
            [f"{city_name}_{osp.basename(frame).split('_')[1]}" for frame in frames_in_city]
        )
        if len(city_seqs) < 10:
            for seq in city_seqs:
                sub_seqs = sorted(
                    glob.glob(osp.join(cityscapes_dir, split, city_name, seq + "*.png"))
                )
                sub_seq_startframe_ids = [
                    osp.basename(sub_seqs[i])[:-16]
                    for i in range(len(sub_seqs))
                    if i % 30 == 0
                ]
                sequences.update(sub_seq_startframe_ids)
        else:
            sequences.update(city_seqs)
    sequence_list = sorted(list(sequences))
    return sequence_list


def main():
    parser = argparse.ArgumentParser(description="Build Cityscapes val sequence list.")
    parser.add_argument(
        "--cityscapes_dir",
        default="",
        help="Root of Cityscapes leftImg8bit_sequence (parent of val/train).",
    )
    parser.add_argument("--split", default="val", help="Dataset split: train/val/test")
    parser.add_argument(
        "--ckpt",
        default="",
    )
    args = parser.parse_args()
    split = "val" if args.split == "test" else args.split
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sequence_list = build_sequence_list(args.cityscapes_dir, split)

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
    abs_rels = []
    delta1s = []
    with torch.no_grad():
        for idx, sequence_name in enumerate(sequence_list):
            
            splits = sequence_name.split("_")
            if len(splits) == 2:  # Sample from Short sequences
                frames_filepaths = sorted(
                    glob.glob(
                        osp.join(
                            args.cityscapes_dir, split, splits[0], sequence_name + "*.png"
                        )
                    )
                )
            elif len(splits) == 3:  # Sample from Long sequences
                frames_filepaths = [
                    osp.join(
                        args.cityscapes_dir,
                        split,
                        splits[0],
                        splits[0]
                        + "_"
                        + splits[1]
                        + "_"
                        + "{:06d}".format(int(splits[2]) + i)
                        + "_leftImg8bit.png",
                    )
                    for i in range(30)
                ]
            else:
                raise ValueError(f"Unexpected sequence_name format: {sequence_name}")


            # Select frames: 14th, 17th, 20th, 23rd (short)
            start_idx = 13  # 0-based index for the 14th frame
            step = 3
            sequence_length = 4
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

            # Use frame 20 (index 19) as GT depth source from VGGT
            gt_path = frames_filepaths[19]
            gt_img = _prepare_frame(gt_path, target_h, target_w)

            images2 = sequence_frames.unsqueeze(0)  # [B,4,3,H,W]
            img12 = images2[:, :2]
            img34 = images2[:, 2:4]

            cond_stage_list, patch_start_idx = model.aggregator.part1(img12)
            tgt_stage_list, _ = model.aggregator.part1(images2)
            x1_big = torch.cat(tgt_stage_list, dim=1)  # [B,4,Ntot,C]
            x1_big = x1_big[:, 2:4, :, :]             # [B,2,Ntot,C]

            B2, T2, Ntot, Ctok = x1_big.shape
            dtype_fm = next(model.fm.parameters()).dtype
            shape_like = torch.zeros((B2, T2, Ntot, Ctok), device=device, dtype=dtype_fm)

            patch_size = model.aggregator.patch_size
            if isinstance(patch_size, (tuple, list)):
                patch_h, patch_w = patch_size
            else:
                patch_h = patch_w = patch_size
            patch_hw = (target_h // patch_h, target_w // patch_w)

            gen_layers = model.fm.sample_euler(
                cond_layers=cond_stage_list,
                shape_like=shape_like,
                steps=50,
                patch_hw=patch_hw,
            )
            gen_tokens = torch.cat(gen_layers, dim=1)  # [B,2,Ntot,C]
            gt_tokens = torch.cat(cond_stage_list, dim=1)[:, 0:2, :, :]  # first two GT frames
            combo_tokens = torch.cat([gt_tokens, gen_tokens], dim=1)     # [B,4,Ntot,C]
            agg_dtype = next(model.aggregator.parameters()).dtype
            combo_stage_list, _ = model.aggregator.part2([combo_tokens.to(agg_dtype)])

            decode_dtype = next(model.depth_head.parameters()).dtype
            combo_stage_list = [x.to(decode_dtype) for x in combo_stage_list]
            images2 = images2.to(decode_dtype)
            depth, _ = model.depth_head(combo_stage_list, images=images2, patch_start_idx=patch_start_idx)
            pred_depth = depth[0, 2, :, :, 0]  # third frame (0-based)

            gt_img = gt_img.to(device)
            gt_out = model.forward_vggt(gt_img)
            gt_depth = gt_out["depth"][0, 0, :, :, 0]

            eps = 1e-6
            valid = (
                (gt_depth > eps)
                & (pred_depth > eps)
                & torch.isfinite(gt_depth)
                & torch.isfinite(pred_depth)
            )
            if torch.any(valid):
                pred_median = torch.median(pred_depth[valid])
                gt_median = torch.median(gt_depth[valid])
                scale = gt_median / torch.clamp(pred_median, min=eps)
                pred_depth_aligned = pred_depth * scale
            else:
                pred_depth_aligned = pred_depth

            abs_rel = torch.mean(
                torch.abs(pred_depth_aligned[valid] - gt_depth[valid]) / torch.clamp(gt_depth[valid], min=eps)
            ).item()
            ratio = torch.maximum(
                pred_depth_aligned / torch.clamp(gt_depth, min=eps),
                gt_depth / torch.clamp(pred_depth_aligned, min=eps),
            )
            delta1 = torch.mean((ratio < 1.25)[valid].float()).item()
            abs_rels.append(abs_rel)
            delta1s.append(delta1)
            print(idx, sequence_name, abs_rel, delta1)

        if abs_rels:
            mean_abs_rel = float(np.mean(abs_rels))
            mean_delta1 = float(np.mean(delta1s))
            print("MEAN", mean_abs_rel, mean_delta1)


if __name__ == "__main__":
    main()
