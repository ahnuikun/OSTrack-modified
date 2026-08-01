"""Paired, read-only response diagnostic for V3 random versus near HNCP.

For every retained training sample this tool uses one clean search crop and one
transformed same-class source instance to create three inputs: clean, V3 random
placement, and nearest valid placement. Random and nearest placements share the
same sampled scale and candidate sequence. The selected checkpoint is evaluated
without gradients; no model, optimizer, config file, or checkpoint is changed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.ostrack.config import cfg as base_cfg
from lib.config.ostrack.config import update_config_from_file
from lib.models.ostrack import build_ostrack
from lib.train.actors import OSTrackActor
from lib.train.admin.settings import Settings
from lib.train.base_functions import build_dataloaders, update_settings
from lib.train.data.vdrm_paired_diagnostics import (
    compute_condition_metrics,
    create_paired_copy_pastes,
    normalized_center_distance,
    summarize_pair_rows,
)


DEFAULT_CONFIG = "vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare clean, V3-random HNCP, and near-target HNCP responses "
            "on exactly paired training samples."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path. Defaults to the selected config's ep0300 "
            "checkpoint below the configured training output directory."
        ),
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--use-lmdb", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for paired_samples.csv and summary.json.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if path.suffix != ".yaml":
        path = PROJECT_ROOT / "experiments" / "ostrack" / f"{path.name}.yaml"
    elif not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    return path


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint(args, settings, config_name: str) -> Path:
    if args.checkpoint is not None:
        path = Path(args.checkpoint).expanduser()
    else:
        path = (
            Path(settings.env.workspace_dir)
            / "checkpoints"
            / "train"
            / "ostrack"
            / config_name
            / "OSTrack_ep0300.pth.tar"
        )
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def _select_tensor(value: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    """Select the collated batch dimension while retaining frame dimension."""
    return value.index_select(1, indexes)


def _metadata_at(value, index: int):
    if isinstance(value, (list, tuple)):
        return value[index]
    if isinstance(value, np.ndarray) and value.ndim:
        return value[index].item() if value[index].ndim == 0 else value[index]
    return value


def _prepare_paired_batch(batch, cfg, remaining: int):
    available = batch["vdrm_diagnostic_distractor_available"][0].bool()
    candidate_indexes = torch.nonzero(available, as_tuple=False).flatten()
    if not candidate_indexes.numel():
        return None

    clean_search = batch["search_images"][0]
    target_boxes = batch["search_anno"][0]
    invalid_masks = batch["search_att"][0].bool()
    source_images = batch["vdrm_diagnostic_distractor_image"][0]
    source_boxes = batch["vdrm_diagnostic_distractor_box"][0]
    min_scale = float(cfg.DATA.SEARCH.VDRM_DISTRACTOR_MIN_SCALE)
    max_scale = float(cfg.DATA.SEARCH.VDRM_DISTRACTOR_MAX_SCALE)

    retained_indexes = []
    random_images = []
    near_images = []
    random_boxes = []
    near_boxes = []
    for raw_index in candidate_indexes.tolist():
        (
            random_image,
            near_image,
            random_applied,
            near_applied,
            random_box,
            near_box,
        ) = create_paired_copy_pastes(
            clean_search[raw_index],
            target_boxes[raw_index],
            source_images[raw_index],
            source_boxes[raw_index],
            invalid_masks[raw_index],
            min_scale,
            max_scale,
        )
        if not (random_applied and near_applied):
            continue
        retained_indexes.append(raw_index)
        random_images.append(random_image)
        near_images.append(near_image)
        random_boxes.append(random_box)
        near_boxes.append(near_box)
        if len(retained_indexes) >= remaining:
            break

    if not retained_indexes:
        return None
    indexes = torch.tensor(retained_indexes, dtype=torch.long)
    return {
        "indexes": indexes,
        "clean_images": clean_search.index_select(0, indexes),
        "random_images": torch.stack(random_images),
        "near_images": torch.stack(near_images),
        "target_boxes": target_boxes.index_select(0, indexes),
        "random_boxes": torch.stack(random_boxes),
        "near_boxes": torch.stack(near_boxes),
    }


def _model_data(batch, indexes, search_images, cfg, device):
    return {
        "template_images": _select_tensor(
            batch["template_images"], indexes
        ).to(device, non_blocking=True),
        "template_anno": _select_tensor(
            batch["template_anno"], indexes
        ).to(device, non_blocking=True),
        "search_images": search_images.unsqueeze(0).to(
            device, non_blocking=True
        ),
        "search_anno": _select_tensor(
            batch["search_anno"], indexes
        ).to(device, non_blocking=True),
        "epoch": int(cfg.TRAIN.EPOCH),
    }


@torch.no_grad()
def _forward_metrics(
    actor,
    data,
    target_boxes,
    distractor_boxes,
    cfg,
):
    output = actor.forward_pass(data)
    metrics = compute_condition_metrics(
        output,
        target_boxes.to(data["search_images"].device),
        search_size=int(cfg.DATA.SEARCH.SIZE),
        stride=int(cfg.MODEL.BACKBONE.STRIDE),
        distractor_boxes=(
            None
            if distractor_boxes is None
            else distractor_boxes.to(data["search_images"].device)
        ),
    )
    return {key: value.detach().cpu() for key, value in metrics.items()}


def _rows_from_batch(
    batch,
    paired,
    clean_random_metrics,
    clean_near_metrics,
    random_metrics,
    near_metrics,
    start_index,
):
    rows = []
    target_boxes = paired["target_boxes"]
    random_boxes = paired["random_boxes"]
    near_boxes = paired["near_boxes"]
    random_distance = normalized_center_distance(target_boxes, random_boxes)
    near_distance = normalized_center_distance(target_boxes, near_boxes)
    indexes = paired["indexes"]

    base_metric_names = (
        "target_logit",
        "target_score",
        "global_negative_logit",
        "global_negative_score",
        "rank_margin",
        "pred_iou",
        "pred_center_error",
        "visual_reliability",
    )
    paste_metric_names = (
        "paste_logit",
        "paste_score",
        "paste_global_gap",
        "paste_hard_hit",
    )
    for local_index, raw_index in enumerate(indexes.tolist()):
        row = {
            "sample_index": start_index + local_index,
            "dataset": str(_metadata_at(batch.get("dataset", ""), raw_index)),
            "object_class": str(
                _metadata_at(batch.get("test_class", ""), raw_index)
            ),
            "target_x": float(target_boxes[local_index, 0]),
            "target_y": float(target_boxes[local_index, 1]),
            "target_w": float(target_boxes[local_index, 2]),
            "target_h": float(target_boxes[local_index, 3]),
            "random_x": float(random_boxes[local_index, 0]),
            "random_y": float(random_boxes[local_index, 1]),
            "random_w": float(random_boxes[local_index, 2]),
            "random_h": float(random_boxes[local_index, 3]),
            "near_x": float(near_boxes[local_index, 0]),
            "near_y": float(near_boxes[local_index, 1]),
            "near_w": float(near_boxes[local_index, 2]),
            "near_h": float(near_boxes[local_index, 3]),
            "random_distance": float(random_distance[local_index]),
            "near_distance": float(near_distance[local_index]),
        }
        for metric in base_metric_names:
            row[f"clean_{metric}"] = float(
                clean_random_metrics[metric][local_index]
            )
            row[f"random_{metric}"] = float(
                random_metrics[metric][local_index]
            )
            row[f"near_{metric}"] = float(
                near_metrics[metric][local_index]
            )
        for metric in paste_metric_names:
            row[f"clean_at_random_{metric}"] = float(
                clean_random_metrics[metric][local_index]
            )
            row[f"clean_at_near_{metric}"] = float(
                clean_near_metrics[metric][local_index]
            )
            row[f"random_{metric}"] = float(
                random_metrics[metric][local_index]
            )
            row[f"near_{metric}"] = float(
                near_metrics[metric][local_index]
            )
        rows.append(row)
    return rows


def _write_outputs(output_dir: Path, rows: list, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_samples.csv"
    json_path = output_dir / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return csv_path, json_path


def _print_summary(summary, csv_path, json_path):
    placement = summary["placement"]
    comparison = summary["comparisons"]["near_minus_random"]
    print("\nPaired HNCP diagnostic completed")
    print(f"samples={summary['sample_count']}")
    print(
        "distance: "
        f"random={placement['random_distance']:.6f} "
        f"near={placement['near_distance']:.6f}"
    )
    print(
        "hard_hit_rate: "
        f"random={placement['random_hard_hit_rate']:.6f} "
        f"near={placement['near_hard_hit_rate']:.6f}"
    )
    print(
        "global_gap: "
        f"random={placement['random_global_gap']:.6f} "
        f"near={placement['near_global_gap']:.6f}"
    )
    print("near_minus_random:")
    for key in (
        "paste_logit",
        "paste_global_gap",
        "paste_hard_hit_rate",
        "target_logit",
        "rank_margin",
        "pred_iou",
        "pred_center_error",
        "visual_reliability",
    ):
        print(f"  {key}={comparison[key]:.6f}")
    print("csv:", csv_path)
    print("summary:", json_path)


def main():
    args = parse_args()
    if (
        args.samples < 1
        or args.max_batches < 1
        or args.batch_size < 1
        or args.workers < 0
    ):
        raise ValueError(
            "samples/max-batches/batch-size must be positive and workers "
            "must be non-negative"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    _set_seed(args.seed)
    config_path = resolve_config_path(args.config)
    config_name = config_path.stem
    cfg = copy.deepcopy(base_cfg)
    update_config_from_file(str(config_path), base_cfg=cfg)
    if not bool(cfg.MODEL.VDRM.ENABLED):
        raise ValueError("paired HNCP diagnostics require VDRM to be enabled")

    # Runtime-only diagnostic overrides. No YAML or training setting is saved.
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.NUM_WORKER = args.workers
    cfg.DATA.SEARCH.VDRM_DISTRACTOR_PROBABILITY = 1.0

    settings = Settings()
    settings.local_rank = -1
    settings.use_lmdb = bool(args.use_lmdb)
    update_settings(settings, cfg)
    checkpoint_path = _resolve_checkpoint(args, settings, config_name)
    loader_train, _ = build_dataloaders(cfg, settings)
    processing = loader_train.dataset.processing
    processing.return_vdrm_diagnostic_inputs = True

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    model = build_ostrack(cfg, training=False)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["net"], strict=True)
    model.to(device).eval()
    actor = OSTrackActor(
        net=model,
        objective={},
        loss_weight={},
        settings=SimpleNamespace(
            batchsize=args.batch_size,
            num_template=int(cfg.DATA.TEMPLATE.NUMBER),
        ),
        cfg=cfg,
    )

    print("config:", config_path)
    print("checkpoint:", checkpoint_path)
    print("device:", device)
    rows = []
    for batch_index, batch in enumerate(loader_train, start=1):
        paired = _prepare_paired_batch(
            batch, cfg, remaining=args.samples - len(rows)
        )
        if paired is not None:
            indexes = paired["indexes"]
            target_boxes = paired["target_boxes"]
            random_boxes = paired["random_boxes"]
            near_boxes = paired["near_boxes"]

            clean_data = _model_data(
                batch, indexes, paired["clean_images"], cfg, device
            )
            clean_output = actor.forward_pass(clean_data)
            clean_random_metrics = compute_condition_metrics(
                clean_output,
                target_boxes.to(device),
                search_size=int(cfg.DATA.SEARCH.SIZE),
                stride=int(cfg.MODEL.BACKBONE.STRIDE),
                distractor_boxes=random_boxes.to(device),
            )
            clean_near_metrics = compute_condition_metrics(
                clean_output,
                target_boxes.to(device),
                search_size=int(cfg.DATA.SEARCH.SIZE),
                stride=int(cfg.MODEL.BACKBONE.STRIDE),
                distractor_boxes=near_boxes.to(device),
            )
            clean_random_metrics = {
                key: value.detach().cpu()
                for key, value in clean_random_metrics.items()
            }
            clean_near_metrics = {
                key: value.detach().cpu()
                for key, value in clean_near_metrics.items()
            }
            del clean_output, clean_data
            random_metrics = _forward_metrics(
                actor,
                _model_data(
                    batch, indexes, paired["random_images"], cfg, device
                ),
                target_boxes,
                random_boxes,
                cfg,
            )
            near_metrics = _forward_metrics(
                actor,
                _model_data(
                    batch, indexes, paired["near_images"], cfg, device
                ),
                target_boxes,
                near_boxes,
                cfg,
            )
            rows.extend(
                _rows_from_batch(
                    batch,
                    paired,
                    clean_random_metrics,
                    clean_near_metrics,
                    random_metrics,
                    near_metrics,
                    start_index=len(rows),
                )
            )
        print(
            f"batch={batch_index}/{args.max_batches} "
            f"paired_samples={len(rows)}/{args.samples}"
        )
        if len(rows) >= args.samples or batch_index >= args.max_batches:
            break

    if not rows:
        raise RuntimeError(
            "no paired samples were produced; inspect class metadata, "
            "dataset paths, and Copy-Paste validity"
        )
    summary = summarize_pair_rows(rows)
    summary.update({
        "config": config_name,
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "requested_samples": args.samples,
        "complete": len(rows) >= args.samples,
        "interpretation": {
            "harder_paste": (
                "Higher paste_logit/hard_hit_rate and lower paste_global_gap."
            ),
            "target_damage": (
                "Lower target_logit, rank_margin, or pred_iou and higher "
                "pred_center_error."
            ),
            "decision_rule": (
                "Near placement is useful only if it is measurably harder "
                "than random without consistently damaging target evidence."
            ),
        },
    })
    dataset_groups = {}
    for dataset_name in sorted({row["dataset"] for row in rows}):
        dataset_rows = [
            row for row in rows if row["dataset"] == dataset_name
        ]
        dataset_groups[dataset_name] = summarize_pair_rows(dataset_rows)
    summary["by_dataset"] = dataset_groups

    if args.output_dir is None:
        output_dir = (
            Path(settings.env.workspace_dir)
            / "vdrm_paired_hncp_diagnostics"
            / config_name
        )
    else:
        output_dir = Path(args.output_dir)
    csv_path, json_path = _write_outputs(output_dir, rows, summary)
    _print_summary(summary, csv_path, json_path)
    if device.type == "cuda":
        print(
            "peak_cuda_memory_gib="
            f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}"
        )


if __name__ == "__main__":
    main()
