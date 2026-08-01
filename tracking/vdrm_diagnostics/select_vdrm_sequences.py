"""Select representative VDRM diagnostic sequences from formal AUC results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.analysis.plot_results import check_and_load_precomputed_results
from lib.test.evaluation import get_dataset, trackerlist


DEFAULT_BASELINE_CONFIG = "vitb_256_mae_ce_32x4_ep300_fulltn"
DEFAULT_VDRM_CONFIG = "vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300"


def select_sequence_groups(
    rows: Iterable[Mapping[str, object]],
    drop_count: int = 10,
    stable_count: int = 5,
    gain_count: int = 5,
):
    """Return disjoint largest-drop, stable, and largest-gain groups."""
    if min(drop_count, stable_count, gain_count) < 0:
        raise ValueError("selection counts must be non-negative")

    normalized = []
    names = set()
    for row in rows:
        name = str(row["sequence"])
        delta = float(row["delta_auc"])
        if not math.isfinite(delta):
            raise ValueError(f"non-finite delta_auc for {name}: {delta}")
        if name in names:
            raise ValueError(f"duplicate sequence: {name}")
        names.add(name)
        normalized.append(dict(row))

    required = drop_count + stable_count + gain_count
    if len(normalized) < required:
        raise ValueError(
            f"need at least {required} sequences, got {len(normalized)}"
        )

    drops = sorted(normalized, key=lambda row: float(row["delta_auc"]))[
        :drop_count
    ]
    gains = sorted(
        normalized,
        key=lambda row: float(row["delta_auc"]),
        reverse=True,
    )[:gain_count]
    used = {str(row["sequence"]) for row in drops + gains}
    stable = sorted(
        [row for row in normalized if str(row["sequence"]) not in used],
        key=lambda row: abs(float(row["delta_auc"])),
    )[:stable_count]

    selected = []
    for category, group in (
        ("largest_drop", drops),
        ("near_unchanged", stable),
        ("largest_gain", gains),
    ):
        selected.extend({**row, "category": category} for row in group)
    return selected


def evaluate_and_select(
    dataset_name: str,
    baseline_config: str,
    vdrm_config: str,
    run_id: Optional[int],
    output_dir: Path,
    drop_count: int,
    stable_count: int,
    gain_count: int,
):
    dataset = get_dataset(dataset_name)
    trackers = trackerlist(
        name="ostrack",
        parameter_name=baseline_config,
        dataset_name=dataset_name,
        run_ids=run_id,
        display_name="Baseline",
    ) + trackerlist(
        name="ostrack",
        parameter_name=vdrm_config,
        dataset_name=dataset_name,
        run_ids=run_id,
        display_name="V3",
    )
    run_label = str(run_id) if run_id is not None else "default"
    evaluation = check_and_load_precomputed_results(
        trackers,
        dataset,
        f"vdrm_selection_{dataset_name}_{run_label}",
        force_evaluation=True,
        skip_missing_seq=False,
    )

    valid = np.asarray(evaluation["valid_sequence"], dtype=bool)
    if not valid.all():
        invalid = [
            name
            for name, is_valid in zip(evaluation["sequences"], valid)
            if not is_valid
        ]
        raise RuntimeError(
            f"invalid or missing formal results for {dataset_name}: {invalid}"
        )

    curves = np.asarray(
        evaluation["ave_success_rate_plot_overlap"], dtype=np.float64
    )
    if curves.ndim != 3 or curves.shape[1] != 2:
        raise RuntimeError(
            "expected [sequence, 2 trackers, threshold] success curves, "
            f"got {curves.shape}"
        )
    per_sequence_auc = curves.mean(axis=2) * 100.0
    overall_auc = curves.mean(axis=0).mean(axis=1) * 100.0

    rows = []
    for index, name in enumerate(evaluation["sequences"]):
        baseline_auc = float(per_sequence_auc[index, 0])
        vdrm_auc = float(per_sequence_auc[index, 1])
        rows.append(
            {
                "sequence": name,
                "baseline_auc": baseline_auc,
                "vdrm_auc": vdrm_auc,
                "delta_auc": vdrm_auc - baseline_auc,
            }
        )
    selected = select_sequence_groups(
        rows,
        drop_count=drop_count,
        stable_count=stable_count,
        gain_count=gain_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    names_path = output_dir / f"{dataset_name}_selected_sequences.txt"
    names_path.write_text(
        "\n".join(str(row["sequence"]) for row in selected) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / f"{dataset_name}_selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "category",
                "sequence",
                "baseline_auc",
                "vdrm_auc",
                "delta_auc",
            ),
        )
        writer.writeheader()
        writer.writerows(selected)

    payload = {
        "dataset": dataset_name,
        "auc_run_id": run_id,
        "overall_auc": {
            "baseline": float(overall_auc[0]),
            "vdrm": float(overall_auc[1]),
            "delta": float(overall_auc[1] - overall_auc[0]),
        },
        "selection": selected,
    }
    json_path = output_dir / f"{dataset_name}_selection.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Select 10/5/5 diagnostic sequences from formal AUC results."
    )
    parser.add_argument(
        "--baseline-config", default=DEFAULT_BASELINE_CONFIG
    )
    parser.add_argument("--vdrm-config", default=DEFAULT_VDRM_CONFIG)
    parser.add_argument(
        "--datasets", nargs="+", default=("uav123", "dtb70")
    )
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drop-count", type=int, default=10)
    parser.add_argument("--stable-count", type=int, default=5)
    parser.add_argument("--gain-count", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    for dataset_name in args.datasets:
        payload = evaluate_and_select(
            dataset_name=dataset_name,
            baseline_config=args.baseline_config,
            vdrm_config=args.vdrm_config,
            run_id=args.run_id,
            output_dir=args.output_dir,
            drop_count=args.drop_count,
            stable_count=args.stable_count,
            gain_count=args.gain_count,
        )
        print(f"\n{dataset_name} formal AUC: {payload['overall_auc']}")
        for row in payload["selection"]:
            print(
                f"{row['category']:14s} {row['sequence']:30s} "
                f"delta_auc={row['delta_auc']:+.4f}"
            )


if __name__ == "__main__":
    main()
