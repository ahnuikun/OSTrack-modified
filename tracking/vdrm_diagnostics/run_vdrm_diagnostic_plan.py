"""Run the registered VDRM backend diagnostic plan end to end."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.evaluation import get_dataset
from tracking.vdrm_diagnostics.diagnose_vdrm_backend_pairs import (
    DEFAULT_BASELINE_CONFIG,
    DEFAULT_VDRM_CONFIG,
    assert_backend_compatibility,
    load_config,
    resolve_config_path,
)


KEY_UAV123_SEQUENCES = (
    "uav_car7",
    "uav_car9",
    "uav_car12",
    "uav_car15",
)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_command(command):
    return " ".join(shlex.quote(str(item)) for item in command)


def run_logged(command, log_path: Path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$", _display_command(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_pair(
    args,
    dataset_name: str,
    sequences,
    anchor_mode: str,
    max_frames: int,
    output_dir: Path,
    cuda_env,
):
    command = [
        sys.executable,
        "-u",
        "tracking/vdrm_diagnostics/diagnose_vdrm_backend_pairs.py",
        "--baseline-config",
        args.baseline_config,
        "--vdrm-config",
        args.vdrm_config,
        "--baseline-checkpoint",
        args.baseline_checkpoint,
        "--vdrm-checkpoint",
        args.vdrm_checkpoint,
        "--dataset-name",
        dataset_name,
        "--sequences",
        *sequences,
        "--anchor-mode",
        anchor_mode,
        "--max-frames",
        str(max_frames),
        "--output-dir",
        output_dir,
        "--device",
        "cuda",
    ]
    run_logged(command, output_dir / "run.log", env=cuda_env)


def preflight(args, run_root: Path, auc_run_id: Optional[int]):
    baseline_checkpoint = Path(args.baseline_checkpoint).expanduser().resolve()
    vdrm_checkpoint = Path(args.vdrm_checkpoint).expanduser().resolve()
    for path in (baseline_checkpoint, vdrm_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_name, baseline_path, baseline_cfg = load_config(
        args.baseline_config
    )
    vdrm_name, vdrm_path, vdrm_cfg = load_config(args.vdrm_config)
    assert_backend_compatibility(baseline_cfg, vdrm_cfg)

    uav_names = {sequence.name for sequence in get_dataset("uav123")}
    missing = sorted(set(KEY_UAV123_SEQUENCES) - uav_names)
    if missing:
        raise RuntimeError(f"missing registered UAV123 sequences: {missing}")
    if not list(get_dataset("dtb70")):
        raise RuntimeError("DTB70 has no registered sequences")

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "baseline_config": baseline_name,
        "baseline_config_path": str(baseline_path),
        "vdrm_config": vdrm_name,
        "vdrm_config_path": str(vdrm_path),
        "baseline_checkpoint": str(baseline_checkpoint),
        "vdrm_checkpoint": str(vdrm_checkpoint),
        "baseline_checkpoint_bytes": baseline_checkpoint.stat().st_size,
        "vdrm_checkpoint_bytes": vdrm_checkpoint.stat().st_size,
        "auc_source": (
            "fresh_explicit_checkpoint"
            if args.run_formal_auc
            else "existing_results"
        ),
        "auc_run_id": auc_run_id,
        "gpu_id": args.gpu_id,
    }
    if not args.skip_checkpoint_hash:
        print("Computing checkpoint SHA-256 hashes...", flush=True)
        manifest["baseline_checkpoint_sha256"] = _sha256(
            baseline_checkpoint
        )
        manifest["vdrm_checkpoint_sha256"] = _sha256(vdrm_checkpoint)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return baseline_checkpoint, vdrm_checkpoint


def run_formal_auc(args, run_root: Path, run_id: int, cuda_env):
    log_dir = run_root / "formal_auc"
    for dataset_name in ("uav123", "dtb70"):
        for label, config, checkpoint in (
            (
                "baseline",
                args.baseline_config,
                args.baseline_checkpoint,
            ),
            ("v3", args.vdrm_config, args.vdrm_checkpoint),
        ):
            command = [
                sys.executable,
                "-u",
                "tracking/test.py",
                "ostrack",
                config,
                "--dataset_name",
                dataset_name,
                "--runid",
                str(run_id),
                "--threads",
                str(args.threads),
                "--num_gpus",
                str(args.num_gpus),
                "--checkpoint",
                checkpoint,
            ]
            log_path = log_dir / f"{dataset_name}_{label}.log"
            run_logged(
                command,
                log_path,
                env=cuda_env,
            )
            expected = f"test checkpoint: {Path(checkpoint).resolve()}"
            if expected not in log_path.read_text(encoding="utf-8"):
                raise RuntimeError(
                    f"formal AUC log did not confirm checkpoint: {expected}"
                )


def select_sequences(args, run_root: Path, run_id: Optional[int]):
    selection_dir = run_root / "selection"
    command = [
        sys.executable,
        "-u",
        "tracking/vdrm_diagnostics/select_vdrm_sequences.py",
        "--baseline-config",
        args.baseline_config,
        "--vdrm-config",
        args.vdrm_config,
        "--datasets",
        "uav123",
        "dtb70",
        "--output-dir",
        selection_dir,
    ]
    if run_id is not None:
        command.extend(("--run-id", str(run_id)))
    run_logged(command, selection_dir / "selection.log")
    selections = {}
    for dataset_name in ("uav123", "dtb70"):
        path = selection_dir / f"{dataset_name}_selected_sequences.txt"
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != 20 or len(set(names)) != 20:
            raise RuntimeError(
                f"expected 20 unique {dataset_name} selections, got {names}"
            )
        selections[dataset_name] = names
    return selections


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run tests, smoke, formal AUC, 10/5/5 selection, four paired "
            "diagnostics, and final root-cause reporting."
        )
    )
    parser.add_argument(
        "--baseline-config", default=DEFAULT_BASELINE_CONFIG
    )
    parser.add_argument("--vdrm-config", default=DEFAULT_VDRM_CONFIG)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--vdrm-checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-formal-auc",
        action="store_true",
        help="Generate fresh formal results with explicit checkpoints.",
    )
    source.add_argument(
        "--use-existing-auc-results",
        action="store_true",
        help="Select from existing tracker results instead of rerunning AUC.",
    )
    parser.add_argument(
        "--auc-run-id",
        type=int,
        default=None,
        help=(
            "Fresh result run ID, or existing result run ID. Omit with "
            "--use-existing-auc-results for the default result directory."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--smoke-frames", type=int, default=10)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="New output directory; defaults to a timestamped directory.",
    )
    parser.add_argument(
        "--full-paired",
        action="store_true",
        help="Also run both anchor modes on every UAV123 and DTB70 sequence.",
    )
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.smoke_frames <= 0:
        raise ValueError("--smoke-frames must be positive")
    if args.run_formal_auc and args.auc_run_id is None:
        args.auc_run_id = int(datetime.now().strftime("%y%m%d%H%M%S"))

    run_root = args.run_root
    if run_root is None:
        run_root = Path("output/vdrm_backend_pairs") / (
            "ep0300_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_root = run_root.resolve()

    baseline_checkpoint, vdrm_checkpoint = preflight(
        args, run_root, args.auc_run_id
    )
    args.baseline_checkpoint = str(baseline_checkpoint)
    args.vdrm_checkpoint = str(vdrm_checkpoint)
    cuda_env = os.environ.copy()
    cuda_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    run_logged(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_vdrm_backend_pairs",
            "tests.test_vdrm_diagnostic_automation",
        ],
        run_root / "unit_test.log",
    )
    smoke_dir = run_root / "smoke" / "uav123_baseline_replay"
    run_pair(
        args,
        "uav123",
        ("uav_car7",),
        "baseline_replay",
        args.smoke_frames,
        smoke_dir,
        cuda_env,
    )
    smoke_log = (smoke_dir / "run.log").read_text(encoding="utf-8")
    expected_lines = (
        f"baseline_config: {resolve_config_path(args.baseline_config)}",
        f"vdrm_config: {resolve_config_path(args.vdrm_config)}",
        f"baseline_checkpoint: {baseline_checkpoint}",
        f"vdrm_checkpoint: {vdrm_checkpoint}",
        "anchor_mode: baseline_replay",
        "device: cuda",
    )
    missing_lines = [line for line in expected_lines if line not in smoke_log]
    if missing_lines:
        raise RuntimeError(f"smoke startup log is missing: {missing_lines}")
    smoke_summary = json.loads(
        (smoke_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    if smoke_summary["frame_count"] != args.smoke_frames:
        raise RuntimeError(
            f"smoke produced {smoke_summary['frame_count']} frames, "
            f"expected {args.smoke_frames}"
        )

    for anchor_mode in ("ground_truth", "baseline_replay"):
        run_pair(
            args,
            "uav123",
            KEY_UAV123_SEQUENCES,
            anchor_mode,
            0,
            run_root / "uav123_key" / anchor_mode,
            cuda_env,
        )

    if args.run_formal_auc:
        run_formal_auc(args, run_root, args.auc_run_id, cuda_env)
    selections = select_sequences(args, run_root, args.auc_run_id)

    for dataset_name in ("uav123", "dtb70"):
        for anchor_mode in ("ground_truth", "baseline_replay"):
            run_pair(
                args,
                dataset_name,
                selections[dataset_name],
                anchor_mode,
                0,
                run_root / dataset_name / anchor_mode,
                cuda_env,
            )

    if args.full_paired:
        for dataset_name in ("uav123", "dtb70"):
            all_sequences = [
                sequence.name for sequence in get_dataset(dataset_name)
            ]
            for anchor_mode in ("ground_truth", "baseline_replay"):
                run_pair(
                    args,
                    dataset_name,
                    all_sequences,
                    anchor_mode,
                    0,
                    run_root / "full" / dataset_name / anchor_mode,
                    cuda_env,
                )

    report_command = [
        sys.executable,
        "-u",
        "tracking/vdrm_diagnostics/summarize_vdrm_diagnostic_plan.py",
        "--run-root",
        run_root,
        "--baseline-checkpoint",
        baseline_checkpoint,
        "--vdrm-checkpoint",
        vdrm_checkpoint,
    ]
    if args.full_paired:
        report_command.append("--prefer-full")
    run_logged(report_command, run_root / "report.log")
    print("\nVDRM diagnostic plan completed")
    print("run_root:", run_root)
    print("report:", run_root / "diagnostic_report.md")


if __name__ == "__main__":
    main()
