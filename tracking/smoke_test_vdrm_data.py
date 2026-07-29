"""Read-only real-dataloader smoke test for VDRM hard distractors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.train.admin.settings import Settings
from lib.train.base_functions import build_dataloaders, update_settings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300",
        help="OSTrack experiment name or YAML path",
    )
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use-lmdb", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if path.suffix != ".yaml":
        path = PROJECT_ROOT / "experiments" / "ostrack" / f"{config_arg}.yaml"
    elif not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    return path


def main():
    args = parse_args()
    if args.batches < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("batches/batch-size must be positive and workers non-negative")

    config_path = resolve_config_path(args.config)
    update_config_from_file(str(config_path))
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_WORKER = args.workers

    probability = float(
        getattr(cfg.DATA.SEARCH, "VDRM_DISTRACTOR_PROBABILITY", 0.0)
    )
    if probability <= 0.0:
        raise ValueError(
            "the selected config does not enable same-class distractors"
        )

    settings = Settings()
    settings.local_rank = -1
    settings.use_lmdb = bool(args.use_lmdb)
    update_settings(settings, cfg)
    loader_train, _ = build_dataloaders(cfg, settings)

    applied = 0
    observed = 0
    check_alignment_boxes = bool(
        getattr(cfg.TRAIN, "VDRM_ALIGN_DISTRACTOR_RANK", False)
    )
    for batch_index, batch in enumerate(loader_train):
        flags = batch["vdrm_distractor_applied"].detach().float()
        if not torch.isfinite(flags).all():
            raise RuntimeError("non-finite distractor flags")
        applied += int(flags.sum().item())
        observed += flags.numel()
        if check_alignment_boxes:
            boxes = batch["vdrm_distractor_box"].detach().float().reshape(-1, 4)
            flat_flags = flags.reshape(-1).bool()
            if boxes.shape[0] != flat_flags.shape[0]:
                raise RuntimeError(
                    "distractor boxes and flags have different batch sizes"
                )
            if not torch.isfinite(boxes).all():
                raise RuntimeError("non-finite distractor boxes")
            applied_boxes = boxes[flat_flags]
            if applied_boxes.numel() and (
                (applied_boxes[:, :2] < 0.0).any()
                or (applied_boxes[:, 2:] <= 0.0).any()
                or (
                    applied_boxes[:, :2] + applied_boxes[:, 2:] > 1.0
                ).any()
            ):
                raise RuntimeError(
                    "applied distractor boxes are outside normalized xywh bounds"
                )
        print(
            f"batch={batch_index + 1} "
            f"search_shape={tuple(batch['search_images'].shape)} "
            f"applied={int(flags.sum().item())}/{flags.numel()}"
        )
        if batch_index + 1 >= args.batches:
            break

    if observed == 0 or applied == 0:
        raise RuntimeError(
            "no same-class distractor was applied; inspect dataset class "
            "metadata and sampling"
        )
    print("VDRM real-data smoke test passed")
    print(f"configured_probability={probability}")
    print(f"observed_applied_rate={applied / observed:.4f}")


if __name__ == "__main__":
    main()
