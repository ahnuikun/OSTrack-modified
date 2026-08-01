import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lib.test.evaluation.tracker import Tracker
from tracking.vdrm_diagnostics.select_vdrm_sequences import (
    select_sequence_groups,
)
from tracking.vdrm_diagnostics.summarize_vdrm_diagnostic_plan import (
    TARGETS,
    build_report,
    write_report,
)


class VDRMDiagnosticAutomationTest(unittest.TestCase):
    def test_sequence_groups_are_disjoint_and_have_registered_sizes(self):
        rows = [
            {"sequence": f"sequence_{index}", "delta_auc": index - 12}
            for index in range(25)
        ]

        selected = select_sequence_groups(rows)

        self.assertEqual(len(selected), 20)
        self.assertEqual(
            len({row["sequence"] for row in selected}), 20
        )
        categories = [row["category"] for row in selected]
        self.assertEqual(categories.count("largest_drop"), 10)
        self.assertEqual(categories.count("near_unchanged"), 5)
        self.assertEqual(categories.count("largest_gain"), 5)
        stable = [
            abs(float(row["delta_auc"]))
            for row in selected
            if row["category"] == "near_unchanged"
        ]
        self.assertLessEqual(max(stable), 2.0)

    def test_sequence_selection_rejects_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate sequence"):
            select_sequence_groups(
                [
                    {"sequence": "same", "delta_auc": -1.0},
                    {"sequence": "same", "delta_auc": 1.0},
                ],
                drop_count=1,
                stable_count=0,
                gain_count=1,
            )

    def test_explicit_checkpoint_overrides_parameter_default(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            tracker = Tracker(
                "fake_tracker",
                "fake_parameters",
                "uav123",
                checkpoint=checkpoint.name,
            )
            module = SimpleNamespace(
                parameters=lambda _: SimpleNamespace(checkpoint="automatic")
            )

            with patch(
                "lib.test.evaluation.tracker.importlib.import_module",
                return_value=module,
            ):
                params = tracker.get_parameters()

            self.assertTrue(
                os.path.samefile(params.checkpoint, checkpoint.name)
            )

    def test_report_validates_four_outputs_and_writes_delivery_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for dataset, anchor_mode in TARGETS:
                directory = root / dataset / anchor_mode
                directory.mkdir(parents=True)
                sequence = f"{dataset}_sequence"
                delta = -0.02 if anchor_mode == "ground_truth" else -0.03
                sequence_summary = {
                    "frame_count": 1,
                    "mean_delta_iou": delta,
                }
                summary = {
                    "dataset": dataset,
                    "anchor_mode": anchor_mode,
                    "sequence_count": 1,
                    "frame_count": 1,
                    "baseline_checkpoint": "baseline.pth.tar",
                    "vdrm_checkpoint": "v3.pth.tar",
                    "mean_delta_iou": delta,
                    "mean_center_only_delta_iou": -0.01,
                    "mean_size_only_delta_iou": -0.005,
                    "catastrophic_vdrm_frames": 0,
                    "sequences": {sequence: sequence_summary},
                    "largest_vdrm_losses": [
                        {"sequence": sequence, "delta_iou": delta}
                    ],
                    "reliability": {
                        "visual_reliability": {
                            "correct_vs_failed_auc": 0.55,
                            "spearman_with_vdrm_iou": 0.1,
                        }
                    },
                    "residual": {
                        "residual_relative_norm_mean": {
                            "correct_mean": 0.1,
                            "failed_mean": 0.2,
                        },
                        "residual_top10_energy_fraction": {
                            "correct_mean": 0.3,
                            "failed_mean": 0.4,
                        },
                    },
                }
                (directory / "dataset_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                (directory / f"{sequence}_summary.json").write_text(
                    json.dumps(sequence_summary), encoding="utf-8"
                )
                with (directory / f"{sequence}.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=("frame",))
                    writer.writeheader()
                    writer.writerow({"frame": 1})

            rows, losses, recommendations = build_report(root)
            outputs = write_report(root, rows, losses, recommendations)

            self.assertEqual(len(rows), 4)
            self.assertEqual(len(losses), 4)
            self.assertEqual(len(recommendations), 2)
            self.assertTrue(all(path.is_file() for path in outputs))


if __name__ == "__main__":
    unittest.main()
