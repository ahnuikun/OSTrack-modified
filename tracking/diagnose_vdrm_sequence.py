"""Export frame-level VDRM reliability diagnostics for selected sequences.

This tool reuses the normal OSTrack inference path. It does not modify model
weights, training settings, tracker state transitions, or standard result
files. One CSV and one JSON summary are written for every selected sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import _init_paths  # noqa: F401

from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.tracker import Tracker


DEFAULT_CONFIG = "vitb_256_mae_ce_vdrm_32x4_ep300"
DEFAULT_SEQUENCES = ("uav_car15", "uav_car7", "uav_car12", "uav_car9")
CORRECT_IOU_THRESHOLD = 0.5
FAILURE_IOU_THRESHOLD = 0.1
SUSTAINED_FAILURE_LENGTH = 5
HIGH_RELIABILITY_THRESHOLD = 0.7
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export per-frame VDRM diagnostics for selected sequences."
    )
    parser.add_argument("--dataset_name", default="uav123")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help="Exact sequence names as exposed by the project dataset adapter.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Output root. Defaults to "
            "<save_dir>/vdrm_diagnostics/<config>/<dataset>."
        ),
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=1,
        help="Token radius suppressed around the first response peak.",
    )
    return parser.parse_args()


def _read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _valid_box(box):
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    return (
        box.size >= 4
        and np.isfinite(box[:4]).all()
        and box[2] > 0.0
        and box[3] > 0.0
    )


def _bbox_iou_xywh(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)[:4]
    second = np.asarray(second, dtype=np.float64).reshape(-1)[:4]
    if not _valid_box(first) or not _valid_box(second):
        return math.nan

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0.0 else math.nan


def response_statistics(response_map, nms_radius=1):
    """Compute peak separation, entropy, and response-only reliability."""
    if nms_radius < 0:
        raise ValueError(f"nms_radius must be non-negative, got {nms_radius}")

    response = torch.as_tensor(response_map).detach().float().squeeze()
    if response.ndim != 2 or response.numel() == 0:
        raise ValueError(
            "response_map must reduce to a non-empty 2-D map, "
            f"got shape {tuple(response.shape)}"
        )

    height, width = response.shape
    flat = response.flatten()
    peak1_index = int(flat.argmax().item())
    peak1_y, peak1_x = divmod(peak1_index, width)
    peak1 = float(flat[peak1_index].item())

    suppressed = response.clone()
    y0 = max(0, peak1_y - nms_radius)
    y1 = min(height, peak1_y + nms_radius + 1)
    x0 = max(0, peak1_x - nms_radius)
    x1 = min(width, peak1_x + nms_radius + 1)
    suppressed[y0:y1, x0:x1] = -torch.inf
    peak2_index = int(suppressed.flatten().argmax().item())
    peak2_y, peak2_x = divmod(peak2_index, width)
    peak2 = float(suppressed.flatten()[peak2_index].item())
    if not math.isfinite(peak2):
        peak2 = 0.0
        peak2_x = -1
        peak2_y = -1

    nonnegative = response.clamp_min(0.0)
    probability = nonnegative.flatten()
    probability_sum = float(probability.sum().item())
    if probability_sum <= EPS:
        entropy_normalized = 1.0
    elif probability.numel() <= 1:
        entropy_normalized = 0.0
    else:
        probability = probability / probability.sum()
        entropy = -(probability * probability.clamp_min(EPS).log()).sum()
        entropy_normalized = float(
            (entropy / math.log(probability.numel())).clamp(0.0, 1.0).item()
        )

    peak_ratio = peak2 / (peak1 + EPS)
    peak_separation = max(0.0, min(1.0, 1.0 - peak_ratio))
    response_reliability = (
        max(0.0, min(1.0, peak1))
        * peak_separation
        * max(0.0, 1.0 - entropy_normalized)
    )
    return {
        "response_p1": peak1,
        "response_p2": peak2,
        "response_peak_ratio": peak_ratio,
        "response_peak_margin": peak1 - peak2,
        "response_entropy_normalized": entropy_normalized,
        "response_reliability": response_reliability,
        "response_p1_x": peak1_x,
        "response_p1_y": peak1_y,
        "response_p2_x": peak2_x,
        "response_p2_y": peak2_y,
    }


def _tracking_status(iou):
    if not math.isfinite(iou):
        return "invalid_gt"
    if iou >= CORRECT_IOU_THRESHOLD:
        return "correct"
    if iou < FAILURE_IOU_THRESHOLD:
        return "failed"
    return "ambiguous"


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(values, targets):
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(targets)
    if valid.sum() < 2:
        return None
    first = _rankdata(values[valid])
    second = _rankdata(targets[valid])
    if np.std(first) <= EPS or np.std(second) <= EPS:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _discrimination_auc(rows, metric):
    scores = []
    labels = []
    for row in rows:
        value = row.get(metric, math.nan)
        status = row["tracking_status"]
        if math.isfinite(value) and status in ("correct", "failed"):
            scores.append(value)
            labels.append(1 if status == "correct" else 0)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _rankdata(scores)
    rank_sum_positive = float(ranks[labels == 1].sum())
    statistic = (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    )
    return statistic / (positive_count * negative_count)


def _first_sustained_failure(rows):
    failed = [row["tracking_status"] == "failed" for row in rows]
    length = SUSTAINED_FAILURE_LENGTH
    for start in range(len(failed) - length + 1):
        if all(failed[start:start + length]):
            return {
                "frame_index": rows[start]["frame_index"],
                "frame_number": rows[start]["frame_number"],
                "image_name": rows[start]["image_name"],
            }
    return None


def summarize_rows(rows):
    valid_rows = [row for row in rows if math.isfinite(row["iou"])]
    correct_rows = [row for row in rows if row["tracking_status"] == "correct"]
    failed_rows = [row for row in rows if row["tracking_status"] == "failed"]
    ambiguous_rows = [
        row for row in rows if row["tracking_status"] == "ambiguous"
    ]

    part_metrics = sorted({
        key
        for row in rows
        for key in row
        if key.startswith("part_") and key.endswith("_reliability")
    })
    reliability_metrics = [
        "visual_reliability",
        "response_reliability",
        "response_p1",
        "response_peak_margin",
        "response_entropy_normalized",
    ] + part_metrics
    metric_summary = {}
    for metric in reliability_metrics:
        metric_summary[metric] = {
            "mean_correct": _mean([
                row.get(metric, math.nan) for row in correct_rows
            ]),
            "mean_failed": _mean([
                row.get(metric, math.nan) for row in failed_rows
            ]),
            "spearman_with_iou": _spearman(
                [row.get(metric, math.nan) for row in valid_rows],
                [row["iou"] for row in valid_rows],
            ),
            "correct_vs_failed_auc": _discrimination_auc(rows, metric),
        }

    high_reliability_failures = sum(
        row["tracking_status"] == "failed"
        and math.isfinite(row["visual_reliability"])
        and row["visual_reliability"] >= HIGH_RELIABILITY_THRESHOLD
        for row in rows
    )
    high_reliability_failure_fraction = (
        high_reliability_failures / len(failed_rows) if failed_rows else None
    )
    return {
        "frame_count": len(rows),
        "valid_gt_frames": len(valid_rows),
        "correct_frames": len(correct_rows),
        "ambiguous_frames": len(ambiguous_rows),
        "failed_frames": len(failed_rows),
        "mean_iou": _mean([row["iou"] for row in valid_rows]),
        "first_sustained_failure": _first_sustained_failure(rows),
        "high_reliability_failed_frames": high_reliability_failures,
        "high_reliability_failure_fraction": (
            high_reliability_failure_fraction
        ),
        "high_reliability_threshold": HIGH_RELIABILITY_THRESHOLD,
        "metrics": metric_summary,
    }


def _base_row(frame_index, frame_path, ground_truth, predicted_box):
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1)[:4]
    predicted_box = np.asarray(predicted_box, dtype=np.float64).reshape(-1)[:4]
    iou = _bbox_iou_xywh(ground_truth, predicted_box)
    return {
        "frame_index": frame_index,
        "frame_number": frame_index + 1,
        "image_name": Path(frame_path).name,
        "gt_valid": _valid_box(ground_truth),
        "gt_x": ground_truth[0],
        "gt_y": ground_truth[1],
        "gt_w": ground_truth[2],
        "gt_h": ground_truth[3],
        "pred_x": predicted_box[0],
        "pred_y": predicted_box[1],
        "pred_w": predicted_box[2],
        "pred_h": predicted_box[3],
        "iou": iou,
        "tracking_status": _tracking_status(iou),
    }


def _diagnostic_row(frame_index, frame_path, ground_truth, tracker_output, elapsed,
                    nms_radius):
    row = _base_row(
        frame_index,
        frame_path,
        ground_truth,
        tracker_output["target_bbox"],
    )
    row["elapsed_seconds"] = elapsed
    row["max_score"] = tracker_output.get("score", math.nan)
    row["visual_reliability"] = tracker_output.get(
        "visual_reliability", math.nan
    )
    row["vdrm_alpha"] = tracker_output.get("vdrm_alpha", math.nan)

    part_reliability = tracker_output.get("part_reliability")
    part_valid = tracker_output.get("part_valid")
    if part_reliability is None or part_valid is None:
        raise RuntimeError(
            "Tracker did not return VDRM part diagnostics. "
            "Use a VDRM-enabled config and checkpoint."
        )
    for index, (reliability, valid) in enumerate(
        zip(part_reliability, part_valid)
    ):
        row[f"part_{index}_reliability"] = reliability
        row[f"part_{index}_valid"] = bool(valid)

    row.update(
        response_statistics(tracker_output["response_map"], nms_radius)
    )
    return row


def _initial_row(sequence):
    ground_truth = sequence.ground_truth_rect[0]
    row = _base_row(
        0,
        sequence.frames[0],
        ground_truth,
        sequence.init_info()["init_bbox"],
    )
    row.update({
        "elapsed_seconds": math.nan,
        "max_score": math.nan,
        "visual_reliability": math.nan,
        "vdrm_alpha": math.nan,
        "response_p1": math.nan,
        "response_p2": math.nan,
        "response_peak_ratio": math.nan,
        "response_peak_margin": math.nan,
        "response_entropy_normalized": math.nan,
        "response_reliability": math.nan,
        "response_p1_x": -1,
        "response_p1_y": -1,
        "response_p2_x": -1,
        "response_p2_y": -1,
    })
    return row


def diagnose_sequence(tracker, sequence, nms_radius):
    image = _read_rgb(sequence.frames[0])
    init_info = sequence.init_info()
    initial_output = tracker.initialize(image, init_info) or {}
    previous_output = OrderedDict(initial_output)
    rows = [_initial_row(sequence)]

    for frame_index, frame_path in enumerate(sequence.frames[1:], start=1):
        image = _read_rgb(frame_path)
        info = sequence.frame_info(frame_index)
        info["previous_output"] = previous_output
        ground_truth = sequence.ground_truth_rect[frame_index]
        info["gt_bbox"] = ground_truth

        start_time = time.perf_counter()
        tracker_output = tracker.track(image, info)
        elapsed = time.perf_counter() - start_time
        previous_output = OrderedDict(tracker_output)
        rows.append(
            _diagnostic_row(
                frame_index,
                frame_path,
                ground_truth,
                tracker_output,
                elapsed,
                nms_radius,
            )
        )
    return rows


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _print_summary(sequence_name, summary, csv_path, json_path):
    print(f"\nSequence: {sequence_name}")
    print(
        "frames={frame_count}, valid_gt={valid_gt_frames}, "
        "correct={correct_frames}, ambiguous={ambiguous_frames}, "
        "failed={failed_frames}, mean_iou={mean_iou:.4f}".format(**summary)
    )
    print("first_sustained_failure:", summary["first_sustained_failure"])
    print(
        "high_reliability_failed_frames: "
        f"{summary['high_reliability_failed_frames']} "
        f"(fraction={summary['high_reliability_failure_fraction']})"
    )
    for metric in ("visual_reliability", "response_reliability"):
        values = summary["metrics"][metric]
        print(
            f"{metric}: correct={values['mean_correct']}, "
            f"failed={values['mean_failed']}, "
            f"spearman_iou={values['spearman_with_iou']}, "
            f"auc={values['correct_vs_failed_auc']}"
        )
    for metric, values in summary["metrics"].items():
        if metric.startswith("part_"):
            print(
                f"{metric}: correct={values['mean_correct']}, "
                f"failed={values['mean_failed']}, "
                f"auc={values['correct_vs_failed_auc']}"
            )
    print("csv:", csv_path)
    print("summary:", json_path)


def main():
    args = parse_args()
    if args.nms_radius < 0:
        raise ValueError("--nms_radius must be non-negative")

    dataset = get_dataset(args.dataset_name)
    sequence_by_name = {sequence.name: sequence for sequence in dataset}
    missing = [name for name in args.sequences if name not in sequence_by_name]
    if missing:
        raise ValueError(
            f"Unknown sequences for {args.dataset_name}: {missing}. "
            "Use the exact names exposed by the dataset adapter."
        )

    tracker_info = Tracker("ostrack", args.config, args.dataset_name, None)
    params = tracker_info.get_parameters()
    params.debug = 0
    tracker = tracker_info.create_tracker(params)

    if args.output_dir is None:
        output_dir = (
            Path(env_settings().save_dir)
            / "vdrm_diagnostics"
            / args.config
            / args.dataset_name
        )
    else:
        output_dir = Path(args.output_dir)

    for sequence_name in args.sequences:
        sequence = sequence_by_name[sequence_name]
        rows = diagnose_sequence(tracker, sequence, args.nms_radius)
        summary = summarize_rows(rows)
        summary.update({
            "dataset": args.dataset_name,
            "sequence": sequence_name,
            "config": args.config,
            "checkpoint": str(params.checkpoint),
            "nms_radius": args.nms_radius,
            "correct_iou_threshold": CORRECT_IOU_THRESHOLD,
            "failure_iou_threshold": FAILURE_IOU_THRESHOLD,
            "sustained_failure_length": SUSTAINED_FAILURE_LENGTH,
        })

        csv_path = output_dir / f"{sequence_name}.csv"
        json_path = output_dir / f"{sequence_name}_summary.json"
        _write_csv(csv_path, rows)
        _write_json(json_path, summary)
        _print_summary(sequence_name, summary, csv_path, json_path)


if __name__ == "__main__":
    main()
