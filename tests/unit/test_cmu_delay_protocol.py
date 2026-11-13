import importlib.util
from pathlib import Path

import pytest


def script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "cmu_delay_noise.py"
    spec = importlib.util.spec_from_file_location("cmu_delay_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delay_grid_lies_on_120hz_source():
    config = script().config_for({"sequence_id": "cmu/13/11", "path": "a.npz",
                                  "sha256": "hash", "start_ns": 0,
                                  "end_ns": 3_449_999_999}, .01,
                                 {"method": "raw", "parameters": {}})
    for time in range(config["start_ns"], config["end_ns"] + 1, config["step_ns"]):
        index = time * 120 // 1_000_000_000
        assert index * 1_000_000_000 // 120 == time
    assert config["delay_ms"] == 100
    assert config["noise_std_m"] == .01


def test_short_reference_rejected():
    with pytest.raises(ValueError, match="window"):
        script().config_for({"start_ns": 0, "end_ns": 600_000_000}, 0,
                            {"method": "raw", "parameters": {}})


def test_six_methods_two_noise_conditions_with_checkpoint_pin(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    cases = script().conditions(tmp_path)
    assert len(cases) == 12
    for noise in (0., .01):
        assert {p["method"] for n, p in cases if n == noise} == {
            "raw", "ema", "one_euro", "kalman", "kalman_cv", "learned"}
    learned = [p for _, p in cases if p["method"] == "learned"]
    assert len(learned[0]["parameters"]["checkpoint_sha256"]) == 64


def test_prefilter_sweep_has_unique_run_ids(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    module = script()
    cases = module.conditions(tmp_path, True)
    assert len(cases) == 8
    entry = {"sequence_id": "cmu/13/11", "path": "a.npz", "sha256": "hash",
             "start_ns": 0, "end_ns": 3_450_000_000}
    configs = [module.config_for(entry, noise, p) for noise, p in cases]
    assert len({c["run_id"] for c in configs}) == 8
    assert {p["parameters"]["history_prefilter_ms"] for _, p in cases} == {0, 10, 25, 50}
