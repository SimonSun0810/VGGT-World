from typing import Any, Dict

import torch


def checkpoint_model_state(ckpt_data: Any) -> Dict[str, torch.Tensor]:
    """Extract the model state dict from a trainer checkpoint payload."""
    if isinstance(ckpt_data, dict) and "model" in ckpt_data:
        return ckpt_data["model"]
    return ckpt_data


def overlay_ema_shadow(
    state: Dict[str, torch.Tensor],
    ckpt_data: Any,
    use_ema: bool = True,
) -> tuple[Dict[str, torch.Tensor], int]:
    """Overlay EMA shadow weights onto *state* when present in *ckpt_data*."""
    ema_used_keys = 0
    if (
        use_ema
        and isinstance(ckpt_data, dict)
        and isinstance(ckpt_data.get("ema"), dict)
        and len(ckpt_data["ema"]) > 0
    ):
        merged = dict(state)
        for k, v in ckpt_data["ema"].items():
            if k in merged:
                merged[k] = v.to(dtype=merged[k].dtype)
                ema_used_keys += 1
        return merged, ema_used_keys
    return state, ema_used_keys
