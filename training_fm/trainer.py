# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers
from train_utils.val_dump import save_depth_png, save_feature, write_ply_xyz


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        self._setup_wandb()

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")
        self._setup_ema()

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_wandb(self) -> None:
        self.wandb_run = None
        if not _WANDB_AVAILABLE:
            wandb_conf = getattr(self.logging_conf, "wandb", None)
            if (
                wandb_conf is not None
                and getattr(wandb_conf, "enabled", False)
                and self.rank == 0
            ):
                logging.warning("wandb is enabled but not installed.")
            return

        wandb_conf = getattr(self.logging_conf, "wandb", None)
        enabled = False
        project = None
        name = None
        if wandb_conf is not None:
            enabled = bool(getattr(wandb_conf, "enabled", False))
            project = getattr(wandb_conf, "project", None)
            name = getattr(wandb_conf, "name", None)

        if not enabled and os.environ.get("WANDB_PROJECT"):
            enabled = True
            project = project or os.environ.get("WANDB_PROJECT")

        if not enabled or self.rank != 0:
            return

        if project is None:
            project = "vggtfm"
        if name is None:
            name = os.path.basename(self.logging_conf.log_dir.rstrip("/"))

        self.wandb_run = wandb.init(
            project=project,
            name=name,
            dir=self.logging_conf.log_dir,
            config={
                "max_epochs": self.max_epochs,
                "accum_steps": self.accum_steps,
                "limit_train_batches": self.limit_train_batches,
                "limit_val_batches": self.limit_val_batches,
            },
        )

    def _finish_wandb(self) -> None:
        if self.wandb_run is not None:
            wandb.finish()

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            # logging.info(f"Loading optimizer state dict (rank {self.rank})")
            # self.optims.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info("Skipping optimizer state loading from checkpoint.")

        # Load training progress
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        if self.ema_enabled and "ema" in checkpoint:
            self._load_ema_state(checkpoint["ema"])

    def _setup_ema(self) -> None:
        """Initializes EMA shadow weights from the current model state.

        NOTE: only shadows parameters belonging to the submodules listed in
        ``optim.ema.module_names`` (default: ["fm"]). This avoids cloning the
        frozen 1B aggregator / heads, which would waste several GB of memory
        and slow down each step / validation.
        """
        ema_conf = getattr(self.optim_conf, "ema", None)
        self.ema_enabled = bool(getattr(ema_conf, "enabled", False)) if ema_conf is not None else False
        self.ema_decay = float(getattr(ema_conf, "decay", 0.9999)) if ema_conf is not None else 0.9999
        self.ema_update_every = int(getattr(ema_conf, "update_every", 1)) if ema_conf is not None else 1
        self.ema_eval_with_ema = bool(getattr(ema_conf, "eval_with_ema", True)) if ema_conf is not None else True
        # Submodule prefixes the EMA should track. Accepts OmegaConf ListConfig or list.
        ema_modules = getattr(ema_conf, "module_names", None) if ema_conf is not None else None
        if ema_modules is None:
            ema_modules = ["fm"]
        self.ema_module_names: List[str] = [str(m) for m in ema_modules]
        # Build set of prefixes like "fm." so lookups are O(1) per key.
        self._ema_key_prefixes: Tuple[str, ...] = tuple(f"{m}." for m in self.ema_module_names)

        self.ema_steps = 0
        self.ema_state = None

        if not self.ema_enabled:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        full_state = model.state_dict()
        
        def _make_shadow(v: torch.Tensor) -> torch.Tensor:
            if torch.is_floating_point(v):
                return v.detach().float().clone()
            return v.detach().clone()

        self.ema_state = {
            k: _make_shadow(v)
            for k, v in full_state.items()
            if self._key_belongs_to_ema(k)
        }
        if len(self.ema_state) == 0:
            logging.warning(
                f"EMA enabled but no parameters matched module_names={self.ema_module_names}. "
                f"Disabling EMA."
            )
            self.ema_enabled = False
            self.ema_state = None
            return

        n_fp32 = sum(1 for t in self.ema_state.values() if t.dtype == torch.float32)
        logging.info(
            f"EMA enabled. decay={self.ema_decay}, update_every={self.ema_update_every}, "
            f"eval_with_ema={self.ema_eval_with_ema}, "
            f"module_names={self.ema_module_names}, "
            f"shadow_tensors={len(self.ema_state)} (fp32={n_fp32})"
        )

    def _key_belongs_to_ema(self, key: str) -> bool:
        """Check whether a state_dict key should be tracked by EMA."""
        return any(key.startswith(p) for p in self._ema_key_prefixes)

    def _load_ema_state(self, ema_state: Dict[str, torch.Tensor]) -> None:
        if not self.ema_enabled or self.ema_state is None:
            return
        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model_state = model.state_dict()
        loaded = 0
        for key, tensor in ema_state.items():
            if key in model_state and self._key_belongs_to_ema(key):
                target = self.ema_state[key]
                self.ema_state[key] = tensor.detach().to(
                    dtype=target.dtype, device=target.device
                ).clone()
                loaded += 1
        logging.info(
            f"Loaded EMA state from checkpoint ({loaded}/{len(self.ema_state)} tensors matched)."
        )

    @torch.no_grad()
    def _update_ema(self) -> None:
        if not self.ema_enabled or self.ema_state is None:
            return
        self.ema_steps += 1
        if self.ema_steps % self.ema_update_every != 0:
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model_state = model.state_dict()
        one_minus_decay = 1.0 - self.ema_decay
        for key, ema_tensor in self.ema_state.items():
            model_tensor = model_state.get(key, None)
            if model_tensor is None:
                continue
            if torch.is_floating_point(model_tensor):
                ema_tensor.mul_(self.ema_decay).add_(
                    model_tensor.detach().float(), alpha=one_minus_decay
                )
            else:
                ema_tensor.copy_(model_tensor.detach())

    @contextlib.contextmanager
    def _ema_eval_context(self):
        if not (self.ema_enabled and self.ema_eval_with_ema and self.ema_state is not None):
            yield
            return

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model_state = model.state_dict()
        backup_state = {
            k: model_state[k].detach().clone()
            for k in self.ema_state.keys()
            if k in model_state
        }
        cast_ema_state = {
            k: (v.to(dtype=model_state[k].dtype) if k in model_state else v)
            for k, v in self.ema_state.items()
        }
        model.load_state_dict(cast_ema_state, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup_state, strict=False)

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            if self.distributed_rank == 0:
                save_path = "trainable_params.txt"
                with open(save_path, "w") as f:
                    for name, p in self.model.named_parameters():
                        if p.requires_grad:
                            f.write(name + "\n")
                print(f"[rank0] saved trainable params to {save_path}")
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        if self.ema_enabled and self.ema_state is not None:
            checkpoint_content["ema"] = {
                key: value.detach().cpu()
                for key, value in self.ema_state.items()
            }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )


    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        # import pdb; pdb.set_trace()
        try:
            if self.mode == "train":
                self.run_train()
                # Optionally run a final validation after all training is done
                self.run_val()
            elif self.mode == "val":
                self.run_val()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        finally:
            self._finish_wandb()

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
                # pass
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        with self._ema_eval_context():
            for data_iter, batch in enumerate(val_loader):
                if data_iter > limit_val_batches:
                    break
                
                # measure data loading time
                data_time.update(time.time() - end)
                data_times.append(data_time.val)
                
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self._process_batch(batch)
                batch = copy_data_to_device(batch, self.device, non_blocking=True)

                amp_type = self.optim_conf.amp.amp_dtype
                assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
                if amp_type == "bfloat16":
                    amp_type = torch.bfloat16
                else:
                    amp_type = torch.float16
                
                # compute output
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=self.optim_conf.amp.enabled,
                        dtype=amp_type,
                    ):
                        out_dir = getattr(self.logging_conf, "val_dump_dir", f"./logs/val/{data_iter}")
                        self._validate_and_dump_batch(batch, step=self.epoch, out_dir=out_dir)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                if torch.cuda.is_available():
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)


        return True

    def _validate_and_dump_batch(self, batch, step: int, out_dir: str):
        """
        batch: dict, must contain:
        - batch["images"]: torch.Tensor [B,T,3,H,W]
        """
        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        model.eval()
        if hasattr(model, "fm") and model.fm is not None:
            model.fm.eval()

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        images2 = batch["images"]  # [B,T,3,H,W]
        assert images2.ndim == 5, f"Expect [B,T,3,H,W], got {images2.shape}"

        B, T, C, H, W = images2.shape
        assert T >= 2, f"Need at least 2 frames, got T={T}"

        img12 = images2[:, 0:2]   # [B,1,3,H,W]
        img34 = images2[:, 2:4]   # [B,1,3,H,W]

        # 1) condition tokens from frame 0
        cond_stage_list, patch_start_idx = model.aggregator.part1(img12)

        # 2) gt tokens from frame 1
        tgt_stage_list, _ = model.aggregator.part1(images2)  
        x1_big = torch.cat(tgt_stage_list, dim=1) 
        x1_big = x1_big[:, 2:4, :, :]           

        B2, T2, Ntot, Ctok = x1_big.shape

        # 3) sample tokens for frame 1
        dtype_fm = next(model.fm.parameters()).dtype if (hasattr(model, "fm") and model.fm is not None) else x1_big.dtype

        cond_z_list = cond_stage_list
        shape_like = torch.zeros((B2, T2, Ntot, Ctok), device=x1_big.device, dtype=dtype_fm)

        patch_size = model.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        patch_hw = (H // patch_h, W // patch_w)

        gen_layers = model.fm.sample_euler(
            cond_layers=cond_z_list,
            shape_like=shape_like,
            steps=self.optim_conf.val_sample_steps if hasattr(self.optim_conf, "val_sample_steps") else 50,
            patch_hw=patch_hw,
        ) 

        gen_tokens = torch.cat(gen_layers, dim=1)  # [B,1,Ntot,C]
        gt_tokens = torch.cat(cond_z_list, dim=1)[:, 0:2, :, :]
        combo_tokens = torch.cat([gt_tokens, gen_tokens], dim=1)     # [B,4,Ntot,C]

        agg_dtype = next(model.aggregator.parameters()).dtype
        combo_stage_list, _ = model.aggregator.part2([combo_tokens.to(agg_dtype)])

        # decide decode dtype
        if getattr(model, "camera_head", None) is not None:
            decode_dtype = next(model.camera_head.parameters()).dtype
        elif getattr(model, "depth_head", None) is not None:
            decode_dtype = next(model.depth_head.parameters()).dtype
        elif getattr(model, "point_head", None) is not None:
            decode_dtype = next(model.point_head.parameters()).dtype
        else:
            decode_dtype = combo_stage_list[0].dtype

        combo_stage_list = [x.to(decode_dtype) for x in combo_stage_list]
        images2 = images2.to(decode_dtype)

        preds = {}

        # depth head
        if getattr(model, "depth_head", None) is not None:
            depth, depth_conf = model.depth_head(combo_stage_list, images=images2, patch_start_idx=patch_start_idx)
            preds["depth"] = depth.float().cpu().numpy()
            preds["depth_conf"] = depth_conf.float().cpu().numpy()

        # point head
        if getattr(model, "point_head", None) is not None:
            pts3d, pts3d_conf = model.point_head(combo_stage_list, images=images2, patch_start_idx=patch_start_idx)
            preds["pts3d"] = pts3d.float().cpu().numpy()
            preds["pts3d_conf"] = pts3d_conf.float().cpu().numpy()

        # 5) dump to disk
        dump_dir = os.path.join(out_dir, f"step_{step:08d}")
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        patch_size = model.aggregator.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size
        grid_hw = (H // patch_h, W // patch_w)

        save_feature(x1_big, gen_tokens, dump_dir, grid_hw=grid_hw)

        if "depth" in preds:
            depth = preds["depth"][0]
            np.save(os.path.join(dump_dir, "depth.npy"), depth)
            d0 = depth[2, :, :, 0]
            save_depth_png(d0, os.path.join(dump_dir, "depth_t0.png"))
            if depth.shape[0] > 1:
                d1 = depth[3, :, :, 0]
                save_depth_png(d1, os.path.join(dump_dir, "depth_t1.png"))

        if "pts3d" in preds:
            pts = preds["pts3d"][0]
            np.save(os.path.join(dump_dir, "pts3d.npy"), pts)
            p0 = pts[2].reshape(-1, 3)
            mask = np.isfinite(p0).all(axis=1)
            p0 = p0[mask]
            if p0.shape[0] > 200000:
                idx = np.random.choice(p0.shape[0], 200000, replace=False)
                p0 = p0[idx]
            write_ply_xyz(p0, os.path.join(dump_dir, "cloud_t0.ply"))
            if pts.shape[0] > 1:
                p1 = pts[3].reshape(-1, 3)
                mask = np.isfinite(p1).all(axis=1)
                p1 = p1[mask]
                if p1.shape[0] > 200000:
                    idx = np.random.choice(p1.shape[0], 200000, replace=False)
                    p1 = p1[idx]
                write_ply_xyz(p1, os.path.join(dump_dir, "cloud_t1.ply"))

        print(f"[VAL] dumped results to: {dump_dir}")

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            mix_progress = float(exact_epoch) / self.max_epochs
            print("mix_progress:", mix_progress)
            self.model.fm_mix_progress = max(0.0, min(1.0, mix_progress))

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()
            self._update_ema()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["train/loss_total"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps

                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        loss, loss_dict = model(batch["images"])
        log_data = {**loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        if (
            phase == "train"
            and self.wandb_run is not None
            and self.steps[phase] % self.logging_conf.log_freq == 0
            and self.rank == 0
        ):
            total_loss = loss_dict.get("total_loss", loss)
            if torch.is_tensor(total_loss):
                total_loss = total_loss.item()
            wandb.log({"train/total_loss": total_loss}, step=self.steps[phase])
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['images'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
                if (
                    phase == "train"
                    and key in ("objective", "loss_objective")
                    and self.wandb_run is not None
                    and step % self.logging_conf.log_freq == 0
                    and self.rank == 0
                ):
                    wandb.log({"train/total_loss": value}, step=step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data

