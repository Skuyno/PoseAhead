"""Confirm the already selected adaptive config; no hyperparameter sweep, training or test."""

import argparse
import json
from pathlib import Path

from cmu_robust_selection import prepare_panel, run_study, write_json

from poseahead.artifacts import sha256_file
from poseahead.evaluation.selection import select_from_batches
from poseahead.processing.config import create_processor, read_processor_config

_STRESS_PROFILES = {
    "sigma0": {"noise_std_m": 0.},
    "sigma25_dropout002": {"noise_std_m": .025, "dropout_probability": .002},
    "sigma25_outlier01_200": {"noise_std_m": .025, "outlier_probability": .01,
                              "outlier_std_m": .200},
    "sigma25_gap150": {"noise_std_m": .025, "gap_period_ms": 3000, "gap_duration_ms": 150},
    "sigma25_gap300": {"noise_std_m": .025, "gap_period_ms": 3000, "gap_duration_ms": 300},
    "sigma25_stride6": {"noise_std_m": .025, "observation_stride": 6},
}


def frozen_candidates(processor_config, pilot_selection):
    """Verify the complete pilot and pin its selected parameters, not its point scores."""
    pilot_selection = Path(pilot_selection).resolve()
    pilot = select_from_batches(pilot_selection.parent / "selection_protocol.json",
                               pilot_selection.parent / "selection_manifest.json")
    if pilot != json.loads(pilot_selection.read_text(encoding="utf-8")):
        raise ValueError("pilot selection report differs from verified pilot artifacts")
    candidate = pilot["candidate_processors"][pilot["selected_candidate_id"]]
    config = read_processor_config(processor_config)
    processor = create_processor(config["method"], config["parameters"])
    if {"method": processor.method, "parameters": processor.parameters} != candidate:
        raise ValueError("confirmation config differs from the previously selected candidate")
    if candidate["method"] != "adaptive" or candidate["parameters"].get(
        "fallback_method"
    ) != "kalman_cv":
        raise ValueError("confirmation requires the selected same-target Kalman adaptive branch")
    return {"kalman_cv": pilot["candidate_processors"]["kalman_cv"], "frozen_adaptive": candidate}


def prepare_confirmation(dataset, processor_config, pilot_selection, output, panel):
    processors = frozen_candidates(processor_config, pilot_selection)
    if panel not in ("gaussian", "stress"):
        raise ValueError("panel must be gaussian or stress")
    study = prepare_panel(
        dataset, processors, output, protocol_id=f"cmu-adaptive-confirmation-{panel}-v1",
        seeds=(42, 43, 44), profiles=_STRESS_PROFILES if panel == "stress" else None,
        checkpoint_selection="frozen earlier pilot-selected candidate; no tuning/training/test",
    )
    study.update(
        panel=panel, processor_config=str(Path(processor_config).resolve()),
        processor_config_sha256=sha256_file(processor_config),
        pilot_selection=str(Path(pilot_selection).resolve()),
        pilot_selection_sha256=sha256_file(pilot_selection),
        limitation="same three repeatedly consulted clips; noise seeds are not new subjects",
        interpretation="descriptive confirmation of frozen config; not held-out superiority",
    )
    write_json(Path(output) / "confirmation_study.json", study)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("processor_config", type=Path)
    parser.add_argument("pilot_selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--panel", choices=("gaussian", "stress"), default="gaussian")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.run_prepared:
        study = json.loads((args.output / "confirmation_study.json").read_text(encoding="utf-8"))
        if (study["panel"] != args.panel
                or sha256_file(args.dataset) != study["dataset_index_sha256"]):
            raise ValueError("prepared confirmation dataset/panel identity differs")
        for field in ("processor_config", "pilot_selection"):
            path = getattr(args, field)
            if (str(path.resolve()) != study[field]
                    or sha256_file(path) != study[f"{field}_sha256"]):
                raise ValueError(f"prepared confirmation {field} changed")
    else:
        study = prepare_confirmation(args.dataset, args.processor_config, args.pilot_selection,
                                     args.output, args.panel)
    count = sum(len(s["cells"]) for s in study["suites"])
    print(f"Frozen {args.panel} confirmation: {count} cells, 3 noise seeds, no tuning", flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
