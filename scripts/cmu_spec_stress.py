"""Predeclare the remaining one-family-at-a-time SPEC E8 validation stress levels."""

import argparse
import json
from pathlib import Path

from cmu_adaptive_recovery import recovery_candidates
from cmu_robust_selection import prepare_panel, run_study, write_json

from poseahead.artifacts import sha256_file

PROFILES = {
    "sigma0": {},
    "outlier01_fixed150": {"outlier_probability": .01, "outlier_magnitude_m": .150},
    "outlier05_fixed150": {"outlier_probability": .05, "outlier_magnitude_m": .150},
    "missing05": {"dropout_probability": .05},
    "missing15": {"dropout_probability": .15},
    "tracking_gap100": {"tracking_gap_period_ms": 3000, "tracking_gap_duration_ms": 100},
    "tracking_gap250": {"tracking_gap_period_ms": 3000, "tracking_gap_duration_ms": 250},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("processor_config", type=Path, help="original pinned pilot config")
    parser.add_argument("pilot_selection", type=Path)
    parser.add_argument("output", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.run_prepared:
        study = json.loads((args.output / "spec_study.json").read_text(encoding="utf-8"))
        if sha256_file(args.dataset) != study["dataset_index_sha256"]:
            raise ValueError("SPEC stress dataset changed")
        for field in ("processor_config", "pilot_selection"):
            path = getattr(args, field)
            if (str(path.resolve()) != study[field]
                    or sha256_file(path) != study[f"{field}_sha256"]):
                raise ValueError(f"SPEC stress {field} changed")
    else:
        old = recovery_candidates(args.processor_config, args.pilot_selection)
        processors = {k: old[k] for k in ("kalman_cv", "partial_recovery")}
        study = prepare_panel(
            args.dataset, processors, args.output, protocol_id="cmu-spec-e8-stress-v1",
            seeds=(42, 43, 44), profiles=PROFILES,
            checkpoint_selection="unchanged selected recovery candidate; no tuning/training/test",
        )
        study.update(
            processor_config=str(args.processor_config.resolve()),
            processor_config_sha256=sha256_file(args.processor_config),
            pilot_selection=str(args.pilot_selection.resolve()),
            pilot_selection_sha256=sha256_file(args.pilot_selection),
            scope="SPEC E8 non-Gaussian families; Gaussian confirmation already archived",
            declaration="one family at a time; outliers fixed150mm; missing/gap confidence zero",
            limitation="severe missing/gap profiles may fail .95 coverage; do not relax/drop them",
        )
        write_json(args.output / "spec_study.json", study)
    print("Frozen SPEC stress: 2 candidates x 7 profiles x 3 seeds x 3 clips = 126 cells",
          flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
