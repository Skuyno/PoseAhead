"""Command-line entry point for PoseAhead."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from poseahead import __version__
from poseahead.config import load_live_config
from poseahead.processing.config import METHODS


def _gpu_description() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _run_doctor() -> int:
    """Print reproducibility-relevant environment information."""
    print(f"poseahead: {__version__}")
    print(f"python: {platform.python_version()}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"gpu: {_gpu_description()}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poseahead",
        description="PoseAhead research and live-runtime tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="print Python, platform, and GPU information")
    gui_parser = subparsers.add_parser("gui", help="open the desktop control panel")
    gui_parser.add_argument("--repository", type=Path, help="PoseAhead checkout directory")
    config_parser = subparsers.add_parser("check-config", help="validate a live JSON config")
    config_parser.add_argument("path")
    import_parser = subparsers.add_parser("import-h36m", help="import a prepared H36M manifest")
    import_parser.add_argument("manifest")
    import_parser.add_argument("output", help="new NPZ recording path")
    cmu_download = subparsers.add_parser("download-cmu", help="download one official CMU trial")
    cmu_download.add_argument("subject")
    cmu_download.add_argument("trial")
    cmu_download.add_argument("output", help="new directory for originals and provenance")
    cmu_import = subparsers.add_parser("import-cmu", help="import an unlabelled CMU pilot")
    cmu_import.add_argument("directory")
    cmu_import.add_argument("output", help="new NPZ recording path")
    inspect_parser = subparsers.add_parser("inspect-recording", help="validate a pose recording")
    inspect_parser.add_argument("path")
    pack_parser = subparsers.add_parser("pack-sample", help="pack ordered PAFR frames into NPZ")
    pack_parser.add_argument("directory")
    pack_parser.add_argument("output")
    pack_parser.add_argument("--sequence-id", required=True)
    pack_parser.add_argument("--clock-domain", required=True)
    process_parser = subparsers.add_parser("process-recording", help="save a separate derived run")
    process_parser.add_argument("path")
    process_parser.add_argument("output")
    process_parser.add_argument("--run-id", required=True)
    methods = process_parser.add_mutually_exclusive_group()
    methods.add_argument("--method", choices=METHODS,
                         default="raw")
    methods.add_argument("--processor-config")
    process_parser.add_argument("--time-constant-ms", type=float, default=50)
    score_parser = subparsers.add_parser("score-recording", help="score against explicit reference")
    score_parser.add_argument("estimates")
    score_parser.add_argument("reference")
    score_parser.add_argument("--reference-interpolation", choices=("exact", "linear"),
                              default="exact")
    score_parser.add_argument("--max-gap-ms", type=int, default=50)
    experiment_parser = subparsers.add_parser("run-experiment", help="run a baseline experiment")
    experiment_parser.add_argument("config")
    experiment_parser.add_argument("output_directory")
    verify_parser = subparsers.add_parser(
        "verify-experiment", help="verify completed artifact hashes",
    )
    verify_parser.add_argument("directory")
    synthetic_parser = subparsers.add_parser(
        "make-synthetic-experiment", help="create sanity inputs",
    )
    synthetic_parser.add_argument("directory")
    batch_parser = subparsers.add_parser("run-batch", help="run a split-checked experiment suite")
    batch_parser.add_argument("manifest")
    batch_parser.add_argument("output_directory")
    verify_batch_parser = subparsers.add_parser(
        "verify-batch", help="verify suite and child artifacts",
    )
    verify_batch_parser.add_argument("directory")
    index_parser = subparsers.add_parser(
        "freeze-dataset", help="pin labelled recording splits/hashes",
    )
    index_parser.add_argument("output")
    index_parser.add_argument("recordings", nargs="+")
    index_verify_parser = subparsers.add_parser(
        "verify-dataset", help="verify pinned dataset index",
    )
    index_verify_parser.add_argument("index")
    windows_parser = subparsers.add_parser(
        "export-windows", help="export split-safe predictor data",
    )
    windows_parser.add_argument("index")
    windows_parser.add_argument("output_directory")
    windows_parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    windows_parser.add_argument("--horizon-ms", type=int, choices=(33, 66, 100, 150), required=True)
    windows_parser.add_argument("--history-frames", type=int, default=10)
    windows_parser.add_argument("--step-ns", type=int, default=20_000_000)
    windows_parser.add_argument("--stride", type=int, default=1)
    windows_parser.add_argument("--shard-size", type=int, default=256)
    windows_parser.add_argument("--history-sampling", choices=("exact", "linear_past"),
                                default="exact")
    windows_parser.add_argument("--max-history-gap-ns", type=int, default=100_000_000)
    verify_windows_parser = subparsers.add_parser("verify-windows", help="verify predictor shards")
    verify_windows_parser.add_argument("directory")
    normalizer_parser = subparsers.add_parser("fit-normalizer", help="fit train-only normalization")
    normalizer_parser.add_argument("directory")
    normalizer_parser.add_argument("output")
    normalizer_parser.add_argument("--minimum-scale-m", type=float, default=.001)
    train_parser = subparsers.add_parser("train-predictor", help="train one fixed-horizon CPU MLP")
    train_parser.add_argument("config")
    train_parser.add_argument("output_directory")
    freeze_parser = subparsers.add_parser(
        "freeze-comparator", help="freeze a validation comparator",
    )
    freeze_parser.add_argument("batch")
    freeze_parser.add_argument("baseline_run_id")
    freeze_parser.add_argument("output")
    compare_parser = subparsers.add_parser("compare-batch", help="paired baseline comparison")
    compare_parser.add_argument("batch")
    compare_parser.add_argument("baseline_run_id")
    compare_parser.add_argument("--frozen-comparator")
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--repetitions", type=int, default=2000)
    compare_parser.add_argument("--output", help="new JSON report path; never overwritten")
    selection_parser = subparsers.add_parser(
        "select-fixed-comparator", help="select from a pinned robust validation panel",
    )
    selection_parser.add_argument("protocol")
    selection_parser.add_argument("manifest")
    selection_parser.add_argument("--output", help="new JSON report path; never overwritten")
    return parser


_ERROR_GROUPS = {
    "Comparison error": ("freeze-comparator", "compare-batch", "select-fixed-comparator"),
    "Training error": ("train-predictor",),
    "Training data error": ("verify-windows", "fit-normalizer"),
    "Window export error": ("export-windows",),
    "Dataset error": ("freeze-dataset", "verify-dataset"),
    "Batch error": ("verify-batch", "run-batch"),
    "Experiment error": ("verify-experiment", "make-synthetic-experiment", "run-experiment"),
    "Scoring error": ("score-recording",),
    "CMU download error": ("download-cmu",),
    "Recording error": (
        "import-h36m", "import-cmu", "inspect-recording", "pack-sample", "process-recording"
    ),
    "Invalid config": ("check-config",),
}
_ERROR_LABELS = {
    command: label for label, commands in _ERROR_GROUPS.items() for command in commands
}


def _run_recording_command(args):
    from poseahead.core import PoseSequence
    from poseahead.processing.config import create_processor, load_processor
    from poseahead.recording import load_recording, replay, save_recording

    if args.command == "import-cmu":
        from poseahead.datasets.cmu import load_cmu

        sequence = load_cmu(args.directory)
        save_recording(args.output, sequence)
    elif args.command == "import-h36m":
        from poseahead.datasets.human36m_loader import load_human36m

        sequence = load_human36m(args.manifest)
        save_recording(args.output, sequence)
    elif args.command == "pack-sample":
        from poseahead.core.serialization import decode_pose_frame

        frames = [decode_pose_frame(path.read_bytes())
                  for path in Path(args.directory).glob("*.pafr")]
        frames.sort(key=lambda frame: frame.frame_id)
        sequence = PoseSequence(args.sequence_id, args.clock_domain, tuple(frames))
        save_recording(args.output, sequence)
    elif args.command == "process-recording":
        if not args.run_id.strip():
            raise ValueError("run-id must be nonempty")
        source = load_recording(args.path)
        processor = (load_processor(args.processor_config) if args.processor_config
                     else create_processor(args.method, {
                         "time_constant_ms": args.time_constant_ms
                     } if args.method == "ema" else {}))
        frames = []
        for observation in replay(source):
            result = processor.process(observation)
            frames.append(replace(result, metadata={
                **result.metadata, "run_id": args.run_id,
                "source_sequence_id": source.sequence_id, "processor": processor.method,
            }))
        sequence = PoseSequence(
            f"{source.sequence_id}/runs/{args.run_id}", source.clock_domain, tuple(frames)
        )
        save_recording(args.output, sequence)
    else:
        sequence = load_recording(args.path)
    return {
        "sequence_id": sequence.sequence_id, "clock_domain": sequence.clock_domain,
        "frames": len(sequence.frames), "duration_ns": sequence.duration_ns,
        "valid_frames": sum(frame.is_valid for frame in sequence.frames),
    }


def _run_json_command(args):
    """Execute a command; reporting and error handling stay at the CLI boundary."""
    if args.command == "select-fixed-comparator":
        from poseahead.evaluation.selection import select_from_batches

        return select_from_batches(args.protocol, args.manifest)
    if args.command in ("freeze-comparator", "compare-batch"):
        from poseahead.evaluation.comparison import compare_batch, freeze_comparator

        if args.command == "freeze-comparator":
            return freeze_comparator(args.batch, args.baseline_run_id, args.output)
        return compare_batch(
            args.batch, args.baseline_run_id, frozen_comparator=args.frozen_comparator,
            seed=args.seed, repetitions=args.repetitions,
        )
    if args.command == "train-predictor":
        from poseahead.prediction.training import train_predictor

        return train_predictor(args.config, args.output_directory)
    if args.command == "verify-windows":
        from poseahead.datasets.window_store import WindowStore

        return WindowStore(args.directory).verify()
    if args.command == "fit-normalizer":
        from poseahead.datasets.normalization import fit_normalizer

        return fit_normalizer(args.directory, args.output, minimum_scale_m=args.minimum_scale_m)
    if args.command == "export-windows":
        from poseahead.datasets.windows import export_training_windows

        return export_training_windows(
            args.index, args.output_directory, split=args.split,
            horizon_ms=args.horizon_ms, history_frames=args.history_frames,
            step_ns=args.step_ns, stride=args.stride, shard_size=args.shard_size,
            history_sampling=args.history_sampling, max_history_gap_ns=args.max_history_gap_ns,
        )
    if args.command in ("freeze-dataset", "verify-dataset"):
        from poseahead.datasets.index import freeze_dataset, verify_dataset

        return (freeze_dataset(args.recordings, args.output)
                if args.command == "freeze-dataset" else verify_dataset(args.index))
    if args.command in ("run-batch", "verify-batch"):
        from poseahead.evaluation.batch import run_batch, verify_batch

        return (run_batch(args.manifest, args.output_directory)
                if args.command == "run-batch" else verify_batch(args.directory))
    if args.command in ("verify-experiment", "make-synthetic-experiment"):
        from poseahead.evaluation.experiment import verify_experiment
        from poseahead.evaluation.synthetic import create_synthetic_experiment

        action = (verify_experiment if args.command == "verify-experiment"
                  else create_synthetic_experiment)
        return action(args.directory)
    if args.command == "run-experiment":
        from poseahead.evaluation.experiment import run_experiment

        return run_experiment(args.config, args.output_directory)
    if args.command == "score-recording":
        from poseahead.evaluation import score_pose_error
        from poseahead.recording import load_recording

        return score_pose_error(
            load_recording(args.estimates), load_recording(args.reference),
            interpolation=args.reference_interpolation, max_gap_ns=args.max_gap_ms * 1_000_000,
        )
    if args.command == "download-cmu":
        from poseahead.datasets.cmu_download import download_trial

        return download_trial(args.subject, args.trial, args.output)
    if args.command in _ERROR_GROUPS["Recording error"]:
        return _run_recording_command(args)
    if args.command == "check-config":
        return load_live_config(args.path).to_dict()
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the PoseAhead command-line interface, keeping optional imports lazy."""
    args = _build_parser().parse_args(argv)
    if args.command == "doctor":
        return _run_doctor()
    if args.command == "gui":
        try:
            from poseahead.desktop.app import run_gui

            return run_gui(args.repository)
        except (ImportError, RuntimeError) as error:
            print(f"Desktop UI error: {error}", file=sys.stderr)
            return 2
    try:
        result = _run_json_command(args)
        rendered = json.dumps(result, indent=2, allow_nan=False)
        if args.command in ("compare-batch", "select-fixed-comparator") and args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(rendered + "\n")
        print(rendered)
    except (ValueError, OSError, ImportError) as error:
        print(f"{_ERROR_LABELS[args.command]}: {error}", file=sys.stderr)
        return 2
    return 0
