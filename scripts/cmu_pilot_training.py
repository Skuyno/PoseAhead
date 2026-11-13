"""Acquire a small fixed CMU pilot and train four exploratory horizon models.

Not a final benchmark: only five source IDs and imbalanced action distributions.
Test clips are prepared/indexed but never scored or used for selection here.
"""

import argparse
import json
from pathlib import Path

from poseahead.datasets.cmu import load_cmu, parse_asf
from poseahead.datasets.cmu_download import download_trial
from poseahead.datasets.cmu_pilot import (
    check_skeleton_partitions,
    prepare_pilot,
    skeleton_fingerprint,
)
from poseahead.datasets.index import freeze_dataset
from poseahead.datasets.normalization import fit_normalizer
from poseahead.datasets.window_store import WindowStore
from poseahead.datasets.windows import export_training_windows
from poseahead.prediction.training import train_predictor
from poseahead.recording import save_recording

TRIALS = (("02", "01"), ("02", "03"), ("02", "04"),
          ("05", "01"), ("05", "02"), ("06", "01"), ("06", "02"),
          ("13", "01"), ("13", "11"), ("13", "29"),
          ("14", "01"), ("14", "06"), ("14", "27"))

# Explicit additional training motions; validation/test selections never change.
EXTRA_TRAIN_TRIALS = (("02", "05"), ("02", "06"), ("02", "10"),
                      ("05", "05"), ("05", "12"),
                      ("06", "06"), ("06", "10"), ("06", "13"), ("06", "15"))


def selected_trials(expanded=False):
    from poseahead.datasets.splits import CMU_SUBJECT_SPLITS

    additions = EXTRA_TRAIN_TRIALS if expanded else ()
    if any(CMU_SUBJECT_SPLITS.get(f"CMU:{subject}") != "train" for subject, _ in additions):
        raise ValueError("expanded trial selection must modify training only")
    result = TRIALS + additions
    if len(set(result)) != len(result):
        raise ValueError("duplicate selected CMU trial")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/cmu/pilot"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--native-history", action="store_true",
                        help="retain 120-Hz observations and sample causal 20-ms histories")
    parser.add_argument("--expanded-training", action="store_true",
                        help="add nine training clips without changing held-out clips")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10000:
        parser.error("epochs must be 1..10000")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sequences, paths, summaries = [], [], []
    for subject, trial in selected_trials(args.expanded_training):
        directory = args.data / f"{subject}_{trial}"
        if not directory.exists():
            download_trial(subject, trial, directory)
        # Existing directories must have a completed, checksum-valid download.
        source = load_cmu(directory)
        skeleton = parse_asf((directory / f"{subject}.asf").read_text())
        sequence = prepare_pilot(source, skeleton_fingerprint(skeleton),
                                 retain_native_rate=args.native_history)
        sequences.append(sequence)
        check_skeleton_partitions(sequences)
        path = output / f"{subject}_{trial}.npz"
        save_recording(path, sequence)
        paths.append(path)
        summary = {"subject": subject, "trial": trial, "source_frames": len(source.frames),
                   "prepared_frames": len(sequence.frames),
                   "split": sequence.frames[0].metadata["split"]}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    index = output / "index.json"
    freeze_dataset(paths, index)
    with (output / "selection.json").open("x", encoding="utf-8") as stream:
        json.dump({"expanded_training": args.expanded_training,
                   "selection_basis": "motion_descriptions_not_test_metrics",
                   "trials": selected_trials(args.expanded_training),
                   "native_history": args.native_history,
                   "epochs": args.epochs}, stream, indent=2)
    # Small CPU reference job; no CUDA initialization or system settings changes.
    import torch

    torch.set_num_threads(2)
    reports = []
    for horizon in (33, 66, 100, 150):
        run = output / f"h{horizon}"
        run.mkdir()
        for split in ("train", "validation"):
            report = export_training_windows(index, run / split, split=split,
                                             horizon_ms=horizon,
                                             step_ns=(20_000_000 if args.native_history
                                                      else 50_000_000),
                                             history_sampling=("linear_past" if args.native_history
                                                               else "exact"))
            WindowStore(run / split).verify()
            print(json.dumps(report), flush=True)
        fit_normalizer(run / "train", run / "normalizer.json")
        config = {"version": 1, "train_windows": "train", "validation_windows": "validation",
                  "normalizer": "normalizer.json", "seed": 42, "epochs": args.epochs,
                  "batch_size": 64, "hidden_size": 64, "learning_rate": .001}
        with (run / "training.json").open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
        report = train_predictor(run / "training.json", run / "model")
        reports.append(report)
        print(json.dumps(report), flush=True)
    with (output / "pilot_report.json").open("x", encoding="utf-8") as stream:
        json.dump({"purpose": "exploratory_cmu_subject_id_pilot_not_final_benchmark",
                   "source_sampling_hz": 120 if args.native_history else 20,
                   "history_step_ms": 20 if args.native_history else 50,
                   "clips": summaries, "training": reports,
                   "test_scored": False,
                   "limitations": ["actor aliases not independently verified",
                                   "skeleton duplicate check is not proof of distinct people",
                                   "small imbalanced action sample", "single seed",
                                   "no webcam detector noise or artificial delay in this run"]},
                  stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
