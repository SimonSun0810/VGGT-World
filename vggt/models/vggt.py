# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import torch.nn.functional as F

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.fm import Flowmatching, FMConfig

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=True,
        fm_train_mode: str = "stage_1",
        fm_pred_steps: int = 50,
        fm_pred_weight: float = 0.1,
    ):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # ---- FM init: 8-layer condition, each layer token dim = embed_dim (1024) ----
        fm_cfg = FMConfig(
            in_dim=embed_dim,
            model_dim=1024,
            depth=8,
            n_heads=16,
            mlp_ratio=4.0,
            attn_drop=0.0,
            proj_drop=0.0,
            n_max=20000,
            t_frames=2,
            # use_patch_pos=True,
        )
        self.fm = Flowmatching(fm_cfg)
        self.fm = self.fm.to(dtype=torch.bfloat16)

        self.fm_train_mode = fm_train_mode
        self.fm_pred_steps = fm_pred_steps
        self.fm_pred_weight = fm_pred_weight
        self.fm_mix_progress = 0.0

    def _forward_stage_2(self, images: torch.Tensor, query_points: torch.Tensor = None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        if images.shape[1] < 5:
            raise ValueError(f"Need at least 5 frames, got {images.shape[1]}")

        # 1) condition tokens from 12 (no grad, used only for pred sampling)
        
        cond_12, _ = self.aggregator.part1(images[:, 0:2])
        cond_12 = [x.detach() for x in cond_12]

        cond_23, _ = self.aggregator.part1(images[:, 1:3])
        cond_23 = [x.detach() for x in cond_23]

        tgt_1234_stage_list, _ = self.aggregator.part1(images[:, 0:4])
        tgt34_layers = [x[:, 2:4, :, :] for x in tgt_1234_stage_list]

        # 2) gt tokens for 34 and 56 (need gradients, avoid seeing future frames)
        tgt12345_stage_list, _ = self.aggregator.part1(images)
        pseudo_gt34_layers = [x[:, 2:4, :, :] for x in tgt12345_stage_list]
        gt45_layers = [x[:, 3:5, :, :] for x in tgt12345_stage_list]

        # 3) infer pred(34) using 12 as condition (detach and store)
        _, _, _, H_img, W_img = images.shape
        patch_size = self.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        patch_hw = (H_img // patch_h, W_img // patch_w)

        dtype_fm = next(self.fm.parameters()).dtype if self.fm is not None else pseudo_gt34_layers[0].dtype
        shape_like = torch.zeros_like(pseudo_gt34_layers[0], dtype=dtype_fm)

        with torch.no_grad():
            cond_stage_list_fm = [x.to(dtype_fm) for x in cond_12]
            pred_layers = self.fm.sample_euler(
                cond_layers=cond_stage_list_fm,
                shape_like=shape_like,
                # steps=self.fm_pred_steps,
                steps=25,
                patch_hw=patch_hw,
            )
        pred34 = torch.cat(pred_layers, dim=1).to(pseudo_gt34_layers[0].dtype).detach()

        # 4) fuse condition for stage-2 training:
        #    use gt for frame-2, only mix frame-3 (gt from cond_23, pred from pred34)
        mix_weight = getattr(self, "fm_mix_progress", None)
        if mix_weight is None:
            mix_weight = float(self.fm_pred_weight)
        mix_weight = max(0.0, min(1.0, float(mix_weight)))
        gt2 = cond_12[0][:, 0:1, :, :]
        gt3 = cond_23[0][:, 1:2, :, :]
        pred3 = pred34[:, 0:1, :, :]

        mix3 = mix_weight * pred3 + (1.0 - mix_weight) * gt3

        # mix3 = pred3
        cond_mix = torch.cat([gt2, mix3], dim=1)
        cond_layers = [cond_mix]

        # 5) train to predict 56 using fused condition
        fm_loss = self.fm.loss_rectified_multilayer(
            x1_layers=gt45_layers,
            cond_layers=cond_layers,
            patch_hw=patch_hw,
        )

        loss_dict = {
            "train/loss_total": fm_loss,
            "train/loss_fm": fm_loss,
        }
        return fm_loss, loss_dict


    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        if self.fm_train_mode == "stage_2":
            return self._forward_stage_2(images, query_points=query_points)

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        img12 = images[:, :2]     # [B,2,3,H,W]
        img34 = images[:, 2:4]    # [B,2,3,H,W]

        cond_stage_list, _ = self.aggregator.part1(img12) 
        cond_stage_list = [x for x in cond_stage_list]

        tgt_stage_list, _ = self.aggregator.part1(images) 
        tgt_stage_list = [x[:,2:4,:,:] for x in tgt_stage_list]
    
        cond_z_list = []
        tgt_z_list = []

        for cond_x, tgt_x in zip(cond_stage_list, tgt_stage_list):
            cond_z_list.append(cond_x)
            tgt_z_list.append(tgt_x)

        # ---- FM loss in z-space ----
        _, _, _, H_img, W_img = images.shape
        patch_size = self.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        patch_hw = (H_img // patch_h, W_img // patch_w)

        fm_loss = self.fm.loss_rectified_multilayer(
            x1_layers=tgt_z_list,
            cond_layers=cond_z_list,
            patch_hw=patch_hw,
        )

        loss = fm_loss
        loss_dict = {
            "train/loss_total": loss,
            "train/loss_fm": fm_loss,
        }

        return loss, loss_dict
            
    def forward_vggt(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions