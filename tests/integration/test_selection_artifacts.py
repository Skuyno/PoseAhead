import json
import subprocess
import sys
from dataclasses import replace

import pytest

from poseahead.artifacts import sha256_file
from poseahead.cli import main
from poseahead.core import PoseSequence
from poseahead.evaluation.batch import run_batch
from poseahead.evaluation.corruption import corruption_settings
from poseahead.evaluation.experiment import load_experiment_config
from poseahead.evaluation.selection import SELECTION_RULE, select_from_batches
from poseahead.evaluation.synthetic import create_synthetic_experiment
from poseahead.recording import load_recording, save_recording


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(params=[False, True], ids=["legacy", "extended-stress"])
def study(tmp_path, request):
    create_synthetic_experiment(tmp_path / "inputs")
    folder = tmp_path / "inputs"
    raw = load_recording(folder / "truth.npz")
    frames = tuple(replace(f, metadata={**f.metadata, "dataset": "Human3.6M",
                                        "subject": "S8", "split": "validation"})
                   for f in raw.frames)
    labelled = PoseSequence("synthetic-validation-only", raw.clock_domain, frames)
    save_recording(folder / "labelled.npz", labelled)
    protocol = {
        "version": 1, "protocol_id": "synthetic-robust-selection",
        "baseline_id": "raw", "clean_profile": "clean", "candidate_ids": ["raw", "ema"],
        "profile_ids": ["clean", "noisy"], "seeds": [42],
        "sequences": [labelled.sequence_id], "rule": dict(SELECTION_RULE),
    }
    write(folder / "protocol.json", protocol)
    declaration = {
        "protocol_sha256": sha256_file(folder / "protocol.json"),
        "candidate_processors": {}, "profile_settings": {}, "evaluation_settings": None,
        "sequence_inputs": {},
    }
    paths, cells = [], {}
    for method in ("raw", "ema"):
        for profile, noise in (("clean", 0.), ("noisy", .01)):
            config = json.loads((folder / f"{method}.json").read_text(encoding="utf-8"))
            run_id = f"{method}-{profile}"
            config.update(observations="labelled.npz", reference="labelled.npz",
                          run_id=run_id, seed=42, noise_std_m=noise)
            if request.param and profile == "noisy":
                config.update(outlier_probability=.01, outlier_std_m=.2,
                              gap_period_ms=1000, gap_duration_ms=150, observation_stride=6)
            path = folder / f"{run_id}.json"
            write(path, config)
            resolved = load_experiment_config(path)
            declaration["candidate_processors"][method] = resolved["processor"]
            declaration["profile_settings"][profile] = corruption_settings(resolved)
            declaration["evaluation_settings"] = {k: resolved[k] for k in (
                "delay_ms", "step_ns", "window_frames", "max_lag_ms",
                "min_velocity_variation_m_s", "min_correlation", "ambiguity_margin",
            )}
            declaration["sequence_inputs"][labelled.sequence_id] = {
                "observations_sha256": sha256_file(folder / "labelled.npz"),
                "reference_sha256": sha256_file(folder / "labelled.npz"),
                "start_ns": resolved["start_ns"], "end_ns": resolved["end_ns"],
                "clock_domain": labelled.clock_domain,
            }
            paths.append(path.name)
            cells[run_id] = {"candidate_id": method, "profile_id": profile}
    write(folder / "declaration.json", declaration)
    write(folder / "suite.json", {
        "version": 1, "batch_id": "synthetic-robust-validation", "split": "validation",
        "experiments": paths,
    })
    run_batch(folder / "suite.json", folder / "batch")
    write(folder / "selection.json", {
        "version": 1, "protocol_sha256": sha256_file(folder / "protocol.json"),
        "declaration": "declaration.json", "declaration_sha256": sha256_file(
            folder / "declaration.json"
        ),
        "batches": [{"directory": "batch", "sha256": sha256_file(
            folder / "batch/batch_completion.json"
        ), "cells": cells}],
    })
    return folder


def test_real_artifacts_can_select_without_loading_models_or_scoring_test(study, capsys):
    result = select_from_batches(study / "protocol.json", study / "selection.json")
    assert result["status"] == "comparator_selected"
    assert result["selected_candidate_id"] == "raw"
    assert result["test_scored"] is False
    assert result["advantage_claim"] == "not_assessed"
    assert result["candidate_processors"]["raw"] == {"method": "raw", "parameters": {}}
    output = study / "report.json"
    args = ["select-fixed-comparator", str(study / "protocol.json"),
            str(study / "selection.json"), "--output", str(output)]
    assert main(args) == 0
    capsys.readouterr()
    assert json.loads(output.read_text(encoding="utf-8")) == result
    before = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize("mutation", ["batch", "protocol", "declaration", "mapping", "duplicate"])
def test_selection_rejects_tampered_provenance_and_incomplete_mapping(study, mutation):
    manifest = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    if mutation in ("batch", "protocol", "declaration"):
        field = {"protocol": "protocol_sha256", "declaration": "declaration_sha256"}
        if mutation == "batch":
            manifest["batches"][0]["sha256"] = "0" * 64
        else:
            manifest[field[mutation]] = "0" * 64
    elif mutation == "mapping":
        manifest["batches"][0]["cells"].pop("raw-clean")
    else:
        manifest["batches"].append(manifest["batches"][0])
    write(study / "selection.json", manifest)
    with pytest.raises(ValueError):
        select_from_batches(study / "protocol.json", study / "selection.json")


def test_declaration_cannot_relabel_a_different_processor_as_the_same_candidate(study):
    manifest = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    manifest["batches"][0]["cells"]["ema-noisy"]["candidate_id"] = "raw"
    write(study / "selection.json", manifest)
    with pytest.raises(ValueError, match="processor changes"):
        select_from_batches(study / "protocol.json", study / "selection.json")


def test_declared_settings_must_match_completed_artifacts(study):
    declaration = json.loads((study / "declaration.json").read_text(encoding="utf-8"))
    declaration["evaluation_settings"]["delay_ms"] = 150
    write(study / "declaration.json", declaration)
    manifest = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    manifest["declaration_sha256"] = sha256_file(study / "declaration.json")
    write(study / "selection.json", manifest)
    with pytest.raises(ValueError, match="predeclared"):
        select_from_batches(study / "protocol.json", study / "selection.json")


@pytest.mark.parametrize("split", ["train", "test", "unlabelled"])
def test_only_validation_batches_can_feed_selection_even_with_valid_hashes(study, split):
    summary_path = study / "batch/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["split"] = split
    write(summary_path, summary)
    completion_path = study / "batch/batch_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["sha256"]["summary.json"] = sha256_file(summary_path)
    write(completion_path, completion)
    manifest = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    manifest["batches"][0]["sha256"] = sha256_file(completion_path)
    write(study / "selection.json", manifest)
    with pytest.raises(ValueError, match="must be validation"):
        select_from_batches(study / "protocol.json", study / "selection.json")


def test_artifact_selector_does_not_import_optional_model_or_camera_libraries(study):
    code = (
        "import sys; from poseahead.evaluation.selection import select_from_batches; "
        "select_from_batches(sys.argv[1], sys.argv[2]); "
        "assert not any(n.startswith(('torch', 'cv2', 'tkinter')) for n in sys.modules)"
    )
    completed = subprocess.run([
        sys.executable, "-c", code, str(study / "protocol.json"), str(study / "selection.json"),
    ], capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
