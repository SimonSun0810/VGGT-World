<div align="center">

# VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model

### Xiangyu Sun, Shijie Wang, Fengyi Zhang, Lin Liu, Caiyan Jia, Ziying Song, Zi Huang, Yadan Luo

[![arXiv](https://img.shields.io/badge/arXiv-2603.12655-b31b1b)](https://arxiv.org/abs/2603.12655)

<img src="assets/pipeline.png" width="95%" alt="VGGT-World main figure">

</div>

This repository contains the training, evaluation, and demo code for **VGGT-World**, built on top of [VGGT](https://github.com/facebookresearch/vggt).

## Installation

**Requirements:** Linux, Python 3.10+, CUDA GPU (tested with CUDA 12.8).

```bash
git clone https://github.com/SimonSun0810/VGGT-World.git
cd VGGT-World

conda create -n vggt_world python=3.10 -y
conda activate vggt_world

pip install -r requirements.txt
```

For other CUDA versions, install `torch` / `torchvision` from [pytorch.org](https://pytorch.org) first, then install the remaining packages.

## Checkpoints

Download our pretrained checkpoints from OneDrive:

**[Checkpoints](https://1drv.ms/f/c/991ebd14386d50f5/IgAJa17DGykcT6LEAhV89viUAYbdSNHZm-woBHrWxHWZYbk?e=XLHhcc)**

After downloading, place the `.pt` file locally, e.g. `checkpoints/kitti_checkpoint.pt`. Checkpoints are PyTorch dicts with a `model` key (online weights).

| Config | Use case |
|--------|----------|
| `default_kitti.yaml` | KITTI short-sequence (stage 1) |
| `default_cityscapes.yaml` | Cityscapes short-sequence (stage 1) |
| `default_cityscapes_stage2.yaml` | Cityscapes mid-sequence (stage 2, autoregressive roll-out) |


## Data Preparation

Please follow [DINO-Foresight](https://github.com/Sta8is/DINO-Foresight) to download and organize the data. Or you can download them [here](https://1drv.ms/f/c/991ebd14386d50f5/IgA0VYpC9c5sQJtzlmQpirC6AfFQgtUhFH_IOe42jTfKm10?e=9Hl92E).

### KITTI

Expected layout (root = `kitti_DIR` in config):

```text
kitti_DIR/
├── train/
│   └── 2011_09_26/2011_09_26_drive_XXXX_sync/image_02/data/*.png
├── val/
│   └── 2011_09_26/2011_09_26_drive_XXXX_sync/image_02/data/*.png
└── val_depth/          # required for eval only
    └── 2011_09_26/2011_09_26_drive_XXXX_sync/proj_depth/groundtruth/image_02/*.png
```

Set the path in config or via CLI override, e.g. `data.train.dataset.dataset_configs.0.kitti_DIR=/path/to/kitti`.

### Cityscapes

Use **leftImg8bit_sequence** (video sequences), not single-frame `leftImg8bit`:

```text
Cityscapes_DIR/
├── train/
│   └── <city>/<city>_<seq>_*_leftImg8bit.png
└── val/
    └── <city>/<city>_<seq>_*_leftImg8bit.png
```

Set `Cityscapes_DIR` in `default_cityscapes.yaml` or override at launch.

## Training

From `training_fm/`:

**Stage 1:**

```bash
python launch.py --config default_cityscapes \
  data.train.dataset.dataset_configs.0.Cityscapes_DIR=/path/to/cityscapes \
  data.val.dataset.dataset_configs.0.Cityscapes_DIR=/path/to/cityscapes \
  checkpoint.resume_checkpoint_path=/path/to/init.pt
```

**Stage 2:**

```bash
python launch.py --config default_cityscapes_stage2 \
  data.train.dataset.dataset_configs.0.Cityscapes_DIR=/path/to/cityscapes \
  checkpoint.resume_checkpoint_path=/path/to/stage1.pt
```


## Evaluation

Run from the **repository root**.

```bash
# KITTI short

python eval/kitti_val_short.py \
  --kitti_root /path/to/kitti \
  --ckpt /path/to/checkpoint.pt
```

```bash
# KITTI mid

python eval/kitti_val_mid.py \
  --kitti_root /path/to/kitti \
  --ckpt /path/to/checkpoint.pt
```

```bash
# Cityscapes short

python eval/cityscapes_val_short.py \
  --cityscapes_dir /path/to/leftImg8bit_sequence \
  --ckpt /path/to/checkpoint.pt
```

```bash
# Cityscapes mid

python eval/cityscapes_val_mid.py \
  --cityscapes_dir /path/to/leftImg8bit_sequence \
  --ckpt /path/to/checkpoint.pt
```

## Demo

A minimal KITTI demo is included with preprocessed sample frames under `demo/frames/` (224×448, same crop as eval).

```bash
# from repo root
python demo/kitti_demo.py \
  --ckpt /path/to/kitti_checkpoint.pt \
```

**Inputs:** `frame1`, `frame2` (conditioning); `frame3`, `frame4` (context for depth decode). Defaults use bundled frames from the first KITTI val sequence.

**Outputs** (color depth maps in `demo/outputs/`):

- `pred_depth_frame3.png`, `pred_depth_frame4.png` — FM prediction
- `vggt_depth_frame3.png`, `vggt_depth_frame4.png` — VGGT baseline on the same frames

Bundled `demo/frames/gt_depth_*.png` are KITTI projected GT depth visualizations for reference.


## License

This project is based on [VGGT](https://github.com/facebookresearch/vggt). This repository retains `LICENSE.txt` from the upstream VGGT repository and complies with its terms for the VGGT-derived portions. New code added in this repository is subject to the same license unless otherwise noted.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sun2026vggtworld,
  title={VGGT-World: Transforming VGGT into an Autoregressive Geometry World Model},
  author={Sun, Xiangyu and Wang, Shijie and Zhang, Fengyi and Liu, Lin and Jia, Caiyan and Song, Ziying and Huang, Zi and Luo, Yadan},
  journal={arXiv preprint arXiv:2603.12655},
  year={2026}
}
```

## Acknowledgements

This codebase builds upon several open-source projects. We thank the authors of:

- [VGGT](https://github.com/facebookresearch/vggt)
- [FLUX](https://github.com/black-forest-labs/flux)
- [JiT](https://github.com/LTH14/JiT)
- [DINO-Foresight](https://github.com/Sta8is/DINO-Foresight)
