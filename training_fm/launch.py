# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help=(
            "Hydra config overrides, e.g. "
            "optim.ema.enabled=false optim.ema.eval_with_ema=false"
        ),
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        overrides = [o for o in (args.overrides or []) if o not in ("--",)]
        cfg = compose(config_name=args.config, overrides=overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


