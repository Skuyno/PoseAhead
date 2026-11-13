"""Predeclare a partial-tracking recovery correction without retuning neural parameters."""

import argparse
import json
from pathlib import Path

from cmu_adaptive_confirmation import _STRESS_PROFILES, frozen_candidates
from cmu_robust_selection import prepare_panel, run_study, write_json

from poseahead.artifacts import sha256_file


def recovery_candidates(processor_config, pilot_selection):
    old = frozen_candidates(processor_config, pilot_selection)
    adaptive = old["frozen_adaptive"]
    return {**old, "partial_recovery": {
        "method": "adaptive", "parameters": {**adaptive["parameters"],
                                              "preserve_partial_baseline": True},
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("processor_config", type=Path)
    parser.add_argument("pilot_selection", type=Path)
    parser.add_argument("output", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.run_prepared:
        study = json.loads((args.output / "recovery_study.json").read_text(encoding="utf-8"))
        if sha256_file(args.dataset) != study["dataset_index_sha256"]:
            raise ValueError("recovery dataset changed")
        for field in ("processor_config", "pilot_selection"):
            path = getattr(args, field)
            if (str(path.resolve()) != study[field]
                    or sha256_file(path) != study[f"{field}_sha256"]):
                raise ValueError(f"recovery {field} changed")
    else:
        study = prepare_panel(
            args.dataset, recovery_candidates(args.processor_config, args.pilot_selection),
            args.output, protocol_id="cmu-adaptive-partial-recovery-v1", seeds=(42, 43, 44),
            profiles={k: _STRESS_PROFILES[k] for k in ("sigma0", "sigma25_dropout002")},
            checkpoint_selection="unchanged pilot parameters; only per-joint recovery differs",
        )
        study.update(
            processor_config=str(args.processor_config.resolve()),
            processor_config_sha256=sha256_file(args.processor_config),
            pilot_selection=str(args.pilot_selection.resolve()),
            pilot_selection_sha256=sha256_file(args.pilot_selection),
            limitation="post-failure corrective validation; not independent/test evidence",
            declaration="partial baseline retains usable joint/root state; neural support clears",
            confidence="missing-joint confidence now zero; same RNG/positions/validity",
        )
        write_json(args.output / "recovery_study.json", study)
    print("Frozen recovery correction: 3 candidates x 2 profiles x 3 seeds x 3 clips = 54 cells",
          flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
