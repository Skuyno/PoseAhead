"""Freeze then run a validation-only 100-ms robustness panel; never retrain or score test."""

import argparse
import json
from pathlib import Path

from poseahead.artifacts import sha256_file
from poseahead.datasets.index import verify_dataset
from poseahead.evaluation.batch import run_batch, verify_batch
from poseahead.evaluation.corruption import CORRUPTION_DEFAULTS, corruption_settings
from poseahead.evaluation.experiment import load_experiment_config
from poseahead.evaluation.selection import (
    SELECTION_RULE,
    select_from_batches,
    validate_selection_protocol,
)

_PROFILES = {f"sigma{millimetres}": {"noise_std_m": millimetres / 1000,
                                     "dropout_probability": 0.}
             for millimetres in (0, 10, 25, 50)}
_SEEDS = (42, 43, 44)
_EVALUATION = {"delay_ms": 100, "step_ns": 25_000_000, "window_frames": 40,
               "max_lag_ms": 200., "min_velocity_variation_m_s": 1e-4,
               "min_correlation": .6, "ambiguity_margin": .02}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def candidates(clean_checkpoint, noisy_checkpoint):
    def learned(path, tau):
        path = Path(path).resolve()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["horizon_ms"] != 100 or manifest["config"]["seed"] != 42:
            raise ValueError("study requires preselected seed-42 100-ms checkpoints")
        return {"method": "learned", "parameters": {
            "checkpoint": str(path), "checkpoint_sha256": sha256_file(path / "manifest.json"),
            "history_sampling": "linear_past", "history_prefilter_ms": tau,
        }}

    return {
        "kalman_cv": {"method": "kalman_cv", "parameters": {"target_offset_ms": 100}},
        "raw": {"method": "raw", "parameters": {}},
        "clean_mlp": learned(clean_checkpoint, 0),
        **{f"noise_mlp_tau{tau}": learned(noisy_checkpoint, tau) for tau in (0, 10, 25, 50)},
    }


def prepare_study(dataset, clean_checkpoint, noisy_checkpoint, output):
    return prepare_panel(dataset, candidates(clean_checkpoint, noisy_checkpoint), output)


def prepare_panel(dataset, processors, output, *, protocol_id="cmu-robust-fixed-selection-v1",
                  seeds=_SEEDS, checkpoint_selection=(
                      "existing checkpoints; no retraining or best-noise-seed selection"),
                  profiles=None, clean_profile="sigma0"):
    """Declare the complete CMU panel before scores; shared by fixed/adaptive pilots."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("study output already exists; choose a new directory")
    profiles = _PROFILES if profiles is None else profiles
    if (not isinstance(profiles, dict) or any(
        not isinstance(value, dict) or value.keys() - CORRUPTION_DEFAULTS.keys()
        for value in profiles.values()
    )):
        raise ValueError("profiles must contain only declared corruption settings")
    profiles = {name: corruption_settings(value) for name, value in profiles.items()}
    dataset_hash = sha256_file(dataset)
    verify_dataset(dataset)
    index = json.loads(dataset.read_text(encoding="utf-8"))
    entries = [e for e in index["recordings"] if e["split"] == "validation"]
    if len(entries) != 3 or {e["sequence_id"] for e in entries} != {
        "cmu/13/01", "cmu/13/11", "cmu/13/29",
    }:
        raise ValueError("study requires the unchanged three expanded-CMU validation clips")
    protocol = {
        "version": 1, "protocol_id": protocol_id,
        "baseline_id": "kalman_cv", "clean_profile": clean_profile,
        "candidate_ids": list(processors), "profile_ids": list(profiles),
        "seeds": list(seeds), "sequences": [e["sequence_id"] for e in entries],
        "rule": dict(SELECTION_RULE),
    }
    validate_selection_protocol(protocol)
    if profiles[clean_profile] != corruption_settings({}):
        raise ValueError("declared clean profile must not corrupt or omit observations")
    for entry in entries:
        end = entry["end_ns"] // _EVALUATION["step_ns"] * _EVALUATION["step_ns"]
        if entry["start_ns"] != 0 or end - 500_000_000 < 39 * _EVALUATION["step_ns"]:
            raise ValueError("validation clip lacks a full post-warmup metric window")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection_protocol.json", protocol)
    declaration = {
        "protocol_sha256": sha256_file(output / "selection_protocol.json"),
        "candidate_processors": {}, "profile_settings": profiles,
        "evaluation_settings": _EVALUATION, "sequence_inputs": {},
    }
    suites = []
    for seed in seeds:
        folder = output / f"seed{seed}"
        folder.mkdir()
        paths, cells = [], {}
        for entry in entries:
            end = entry["end_ns"] // _EVALUATION["step_ns"] * _EVALUATION["step_ns"]
            declaration["sequence_inputs"][entry["sequence_id"]] = {
                "observations_sha256": entry["sha256"], "reference_sha256": entry["sha256"],
                "start_ns": 500_000_000, "end_ns": end, "clock_domain": entry["clock_domain"],
            }
            for candidate_id, processor in processors.items():
                for profile_id, stress in profiles.items():
                    sequence_id = entry["sequence_id"].replace("/", "-")
                    run_id = f"{sequence_id}-{candidate_id}-{profile_id}-s{seed}"
                    config = {
                        "version": 1, "run_id": run_id,
                        "observations": entry["path"], "reference": entry["path"],
                        "observations_sha256": entry["sha256"], "reference_sha256": entry["sha256"],
                        "processor": processor, "seed": seed, "start_ns": 500_000_000,
                        "end_ns": end, **_EVALUATION, **stress,
                    }
                    path = folder / f"config-{len(paths):03d}.json"
                    write_json(path, config)
                    resolved = load_experiment_config(path)
                    declaration["candidate_processors"][candidate_id] = resolved["processor"]
                    paths.append(path.name)
                    cells[run_id] = {"candidate_id": candidate_id, "profile_id": profile_id}
        write_json(folder / "suite.json", {
            "version": 1, "batch_id": f"cmu-robust-validation-seed{seed}",
            "split": "validation", "experiments": paths,
        })
        suites.append({"directory": folder.name, "cells": cells})
    if sha256_file(dataset) != dataset_hash:
        raise ValueError("dataset changed during study preparation")
    write_json(output / "declaration.json", declaration)
    study = {
        "dataset_index_sha256": dataset_hash, "test_scored": False,
        "protocol_sha256": declaration["protocol_sha256"],
        "declaration_sha256": sha256_file(output / "declaration.json"), "suites": suites,
        "scope": "prospective continuation of a repeatedly consulted validation pilot",
        "corruption": ("synthetic declared observation stress; not calibrated camera error"
                       if profiles != _PROFILES else
                       "independent joint Gaussian only; not calibrated camera error"),
        "checkpoint_selection": checkpoint_selection,
    }
    write_json(output / "study.json", study)
    return study


def run_study(output, study):
    output = Path(output)
    manifest = {
        "version": 1, "protocol_sha256": study["protocol_sha256"],
        "declaration": "declaration.json", "declaration_sha256": study["declaration_sha256"],
        "batches": [],
    }
    for suite in study["suites"]:
        folder = output / suite["directory"]
        print(f"Running {folder.name}: {len(suite['cells'])} validation cells", flush=True)
        run_batch(folder / "suite.json", folder / "batch")
        verified = verify_batch(folder / "batch")
        print(json.dumps(verified), flush=True)
        manifest["batches"].append({
            "directory": f"{folder.name}/batch",
            "sha256": sha256_file(folder / "batch/batch_completion.json"), "cells": suite["cells"],
        })
    write_json(output / "selection_manifest.json", manifest)
    result = select_from_batches(
        output / "selection_protocol.json", output / "selection_manifest.json"
    )
    write_json(output / "selection.json", result)
    print(json.dumps({"status": result["status"],
                      "selected_candidate_id": result["selected_candidate_id"],
                      "advantage_claim": result["advantage_claim"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("clean_checkpoint", type=Path)
    parser.add_argument("noisy_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-prepared", action="store_true",
                        help="run frozen study; completed batches are never overwritten")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.run_prepared and args.prepare_only:
        parser.error("choose prepare-only or run-prepared, not both")
    if args.run_prepared:
        study = json.loads((args.output / "study.json").read_text(encoding="utf-8"))
        if sha256_file(args.dataset) != study["dataset_index_sha256"]:
            raise ValueError("prepared study dataset index changed")
        declaration = json.loads((args.output / "declaration.json").read_text(encoding="utf-8"))
        for name, path in (("clean_mlp", args.clean_checkpoint),
                           ("noise_mlp_tau0", args.noisy_checkpoint)):
            if str(path.resolve()) != declaration["candidate_processors"][name]["parameters"][
                "checkpoint"
            ]:
                raise ValueError("prepared study checkpoint differs from requested path")
    else:
        study = prepare_study(
            args.dataset, args.clean_checkpoint, args.noisy_checkpoint, args.output
        )
    print("Frozen study: 7 candidates x 4 profiles x 3 noise seeds x 3 validation clips",
          flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
