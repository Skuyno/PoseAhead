import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.datasets.index import freeze_dataset
from poseahead.recording import save_recording


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return (importlib.import_module("cmu_robust_selection"),
            importlib.import_module("cmu_adaptive_confirmation"))


@pytest.fixture
def dataset(tmp_path):
    recordings = []
    for trial in ("01", "11", "29"):
        sequence = PoseSequence(f"cmu/13/{trial}", f"cmu-clock-{trial}", tuple(
            PoseFrame(i, i * 25_000_000, i * 25_000_000, np.zeros((17, 3)),
                      np.ones(17), np.ones(17, bool), metadata={
                          "subject": "CMU:13", "dataset": "CMU", "split": "validation",
                          "skeleton_fingerprint": "a" * 64,
                      }) for i in range(61)
        ))
        path = tmp_path / f"{trial}.npz"
        save_recording(path, sequence)
        recordings.append(path)
    index = tmp_path / "index.json"
    freeze_dataset(recordings, index)
    return index


def processors():
    return {"kalman_cv": {"method": "kalman_cv", "parameters": {"target_offset_ms": 100}},
            "raw": {"method": "raw", "parameters": {}}}


def test_generic_panel_predeclares_complete_multiseed_extended_stress(modules, dataset, tmp_path):
    runner, confirmation = modules
    output = tmp_path / "study"
    study = runner.prepare_panel(dataset, processors(), output, seeds=(42, 43, 44),
                                 profiles=confirmation._STRESS_PROFILES)
    assert not (output / "selection.json").exists()
    assert sum(len(s["cells"]) for s in study["suites"]) == 108
    assert study["test_scored"] is False
    declaration = json.loads((output / "declaration.json").read_text())
    assert declaration["profile_settings"]["sigma25_stride6"]["observation_stride"] == 6
    assert declaration["profile_settings"]["sigma0"] == {
        "noise_std_m": 0., "dropout_probability": 0.,
    }
    for seed in (42, 43, 44):
        configs = list((output / f"seed{seed}").glob("config-*.json"))
        assert len(configs) == 36
        assert {json.loads(p.read_text())["seed"] for p in configs} == {seed}
    before = (output / "declaration.json").read_bytes()
    with pytest.raises(FileExistsError):
        runner.prepare_panel(dataset, processors(), output)
    assert (output / "declaration.json").read_bytes() == before


@pytest.mark.parametrize("profiles, seeds", [
    ({"sigma0": {}, "bad": {"unknown": 1}}, (42,)),
    ({"sigma0": {"observation_stride": 6}, "noisy": {"noise_std_m": .01}}, (42,)),
    ({"sigma0": {}, "bad": {"gap_period_ms": 1000}}, (42,)),
    ({"sigma0": {}}, (42,)), ({"sigma0": {}, "noisy": {}}, (42, 42)),
])
def test_invalid_panel_is_rejected_before_output_publication(
    modules, dataset, tmp_path, profiles, seeds,
):
    runner, _ = modules
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        runner.prepare_panel(dataset, processors(), output, profiles=profiles, seeds=seeds)
    assert not output.exists()


def test_default_panel_remains_the_original_gaussian_protocol(modules, dataset, tmp_path):
    runner, _ = modules
    output = tmp_path / "legacy"
    study = runner.prepare_panel(dataset, processors(), output)
    protocol = json.loads((output / "selection_protocol.json").read_text())
    assert protocol["profile_ids"] == ["sigma0", "sigma10", "sigma25", "sigma50"]
    assert protocol["seeds"] == [42, 43, 44]
    assert sum(len(s["cells"]) for s in study["suites"]) == 72


def test_frozen_candidate_must_match_both_verified_report_and_processor(modules, tmp_path,
                                                                       monkeypatch):
    _, confirmation = modules
    params = {"fallback_method": "kalman_cv", "maximum_neural_weight": .25}
    pilot = {"selected_candidate_id": "selected", "candidate_processors": {
        **processors(), "selected": {"method": "adaptive", "parameters": params},
    }}
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(pilot))
    monkeypatch.setattr(confirmation, "select_from_batches", lambda *a: pilot)
    monkeypatch.setattr(confirmation, "read_processor_config", lambda *a: {
        "method": "adaptive", "parameters": params,
    })
    monkeypatch.setattr(confirmation, "create_processor", lambda *a: SimpleNamespace(
        method="adaptive", parameters=params,
    ))
    result = confirmation.frozen_candidates(tmp_path / "config.json", selection)
    assert list(result) == ["kalman_cv", "frozen_adaptive"]
    monkeypatch.setattr(confirmation, "create_processor", lambda *a: SimpleNamespace(
        method="adaptive", parameters={**params, "maximum_neural_weight": .1},
    ))
    with pytest.raises(ValueError, match="previously selected"):
        confirmation.frozen_candidates(tmp_path / "config.json", selection)
    selection.write_text(json.dumps({**pilot, "selected_candidate_id": "raw"}))
    with pytest.raises(ValueError, match="verified pilot"):
        confirmation.frozen_candidates(tmp_path / "config.json", selection)


def test_recovery_ablation_changes_only_partial_baseline_state_policy(modules, monkeypatch):
    recovery = importlib.import_module("cmu_adaptive_recovery")
    params = {"fallback_method": "kalman_cv", "maximum_neural_weight": .25,
              "policy": {"max_horizon_ms": 100}}
    old = {"kalman_cv": processors()["kalman_cv"],
           "frozen_adaptive": {"method": "adaptive", "parameters": params}}
    monkeypatch.setattr(recovery, "frozen_candidates", lambda *a: old)
    result = recovery.recovery_candidates("config", "selection")
    assert result["frozen_adaptive"] == old["frozen_adaptive"]
    assert result["partial_recovery"]["parameters"] == {
        **params, "preserve_partial_baseline": True,
    }
    assert "preserve_partial_baseline" not in params


def test_spec_profiles_match_declared_levels_without_cartesian_gaussian_noise(modules):
    spec = importlib.import_module("cmu_spec_stress")
    assert len(spec.PROFILES) == 7
    for name in ("outlier01_fixed150", "outlier05_fixed150"):
        assert spec.PROFILES[name]["outlier_magnitude_m"] == .150
        assert "outlier_std_m" not in spec.PROFILES[name]
    assert spec.PROFILES["missing05"]["dropout_probability"] == .05
    assert spec.PROFILES["missing15"]["dropout_probability"] == .15
    assert spec.PROFILES["tracking_gap100"]["tracking_gap_duration_ms"] == 100
    assert spec.PROFILES["tracking_gap250"]["tracking_gap_duration_ms"] == 250
    assert all(p.get("noise_std_m", 0) == 0 for p in spec.PROFILES.values())
