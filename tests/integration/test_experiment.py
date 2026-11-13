import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from poseahead.cli import main
from poseahead.core import PoseFrame, PoseSequence
from poseahead.evaluation.experiment import (
    degrade_observations,
    run_experiment,
    verify_experiment,
)
from poseahead.recording import load_recording, save_recording


def setup_experiment(tmp_path, **overrides):
    frames = []
    for i in range(400):
        t = i / 100
        value = (np.exp(-((t - 1.2) / .13) ** 2)
                 - .7 * np.exp(-((t - 2.1) / .21) ** 2)
                 + .4 * np.exp(-((t - 3.2) / .1) ** 2))
        positions = np.zeros((17, 3))
        positions[1:, 0] = value
        frames.append(PoseFrame(i, i * 10_000_000, i * 10_000_000, positions,
                                np.ones(17), np.ones(17, bool), root_position=[value, 0, 3]))
    save_recording(tmp_path / "truth.npz", PoseSequence("pulses", "synthetic", tuple(frames)))
    config = {"version": 1, "run_id": "pulses-raw-100", "observations": "truth.npz",
              "reference": "truth.npz", "processor": {"method": "raw", "parameters": {}},
              "seed": 42, "delay_ms": 100, "start_ns": 200_000_000,
              "end_ns": 3_990_000_000, "step_ns": 10_000_000, "window_frames": 380,
              **overrides}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_complete_experiment_artifacts_replay_and_delay(tmp_path):
    config = setup_experiment(tmp_path)
    raw = (tmp_path / "truth.npz").read_bytes()
    out = tmp_path / "run"
    report = run_experiment(config, out)
    assert report["latency"]["effective_latency_ms"] == pytest.approx(100, abs=.3)
    assert report["latency"]["eligible_joint_windows"] == 16
    assert report["target_error"]["mpjpe_mm"] == 0
    assert report["display_error"]["mpjpe_mm"] > 0
    assert report["display_error"]["root_error_mm"] > 0
    assert report["runtime"]["processor_calls"] == 390
    completion = json.loads((out / "completion.json").read_text())
    for filename, digest in completion["sha256"].items():
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == digest
    again = run_experiment(out / "resolved_config.json", tmp_path / "rerun")
    assert {k: v for k, v in report.items() if k != "runtime"} == {
        k: v for k, v in again.items() if k != "runtime"
    }
    assert (tmp_path / "truth.npz").read_bytes() == raw
    with pytest.raises(FileExistsError):
        run_experiment(config, out)
    resolved = json.loads((out / "resolved_config.json").read_text())
    assert Path(resolved["observations"]).is_absolute()
    assert resolved["noise_std_m"] == 0


@pytest.mark.parametrize("method", ["ema", "one_euro", "kalman", "kalman_cv"])
def test_all_baselines_run_from_config(tmp_path, method):
    config = setup_experiment(tmp_path, processor={"method": method, "parameters": {}})
    out = tmp_path / "run"
    report = run_experiment(config, out)
    assert report["runtime"]["processor_calls"] == 390
    resolved = json.loads((out / "resolved_config.json").read_text())
    assert resolved["processor"]["parameters"]
    assert load_recording(out / "processed.npz").frames[-1].metadata["processor"] == method


def test_noise_seed_repeatability_dropout_and_pelvis_invariant(tmp_path):
    setup_experiment(tmp_path)
    sequence = load_recording(tmp_path / "truth.npz")
    a, b = [degrade_observations(sequence, seed=42, noise_std_m=.01, dropout_probability=.1)
            for _ in range(2)]
    different = degrade_observations(sequence, seed=43, noise_std_m=.01, dropout_probability=.1)
    for left, right in zip(a.frames, b.frames, strict=True):
        np.testing.assert_array_equal(left.joint_positions, right.joint_positions)
        np.testing.assert_array_equal(left.joint_validity, right.joint_validity)
        np.testing.assert_array_equal(left.joint_positions[0], [0, 0, 0])
    assert not np.array_equal(a.frames[0].joint_positions, different.frames[0].joint_positions)
    assert not all(f.joint_validity.all() for f in a.frames)


@pytest.mark.parametrize("override", [
    {"version": True}, {"seed": True}, {"noise_std_m": -1}, {"dropout_probability": 2},
    {"window_frames": 10}, {"step_ns": 0}, {"unknown": 1}, {"max_lag_ms": float("nan")},
    {"observations_sha256": "wrong"}, {"processor": {"method": "raw", "parameters": None}},
    {"start_ns": 201_000_000},
])
def test_invalid_experiment_creates_no_artifacts(tmp_path, override):
    config = setup_experiment(tmp_path, **override)
    with pytest.raises(ValueError):
        run_experiment(config, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_cli_and_missing_observation_coverage(tmp_path, capsys):
    config = setup_experiment(tmp_path, delay_ms=10000)
    assert main(["run-experiment", str(config), str(tmp_path / "run")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["display_error"]["mpjpe_mm"] is None
    assert report["display_error"]["joint_coverage"] == 0
    assert report["target_error"] is None
    assert report["latency"]["effective_latency_ms"] is None
    assert main(["run-experiment", str(config), str(tmp_path / "run")]) == 2
    assert "Experiment error" in capsys.readouterr().err


def test_artifact_verifier_detects_modification_and_path_injection(tmp_path):
    config = setup_experiment(tmp_path)
    out = tmp_path / "run"
    run_experiment(config, out)
    assert verify_experiment(out)["verified_files"] == 6
    metrics = out / "metrics.json"
    original = metrics.read_bytes()
    metrics.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="checksum"):
        verify_experiment(out)
    metrics.write_bytes(original)
    completion = out / "completion.json"
    manifest = json.loads(completion.read_text())
    manifest["sha256"]["../outside"] = "fake"
    completion.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="names"):
        verify_experiment(out)


def test_resolved_config_detects_changed_original_input(tmp_path):
    config = setup_experiment(tmp_path)
    out = tmp_path / "run"
    run_experiment(config, out)
    source = tmp_path / "truth.npz"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        run_experiment(out / "resolved_config.json", tmp_path / "rerun")
    assert not (tmp_path / "rerun").exists()


def test_failure_during_publication_leaves_no_completion_marker(tmp_path, monkeypatch):
    import poseahead.evaluation.experiment as experiment

    config = setup_experiment(tmp_path)
    def fail(*args):
        raise OSError("simulated full disk")
    monkeypatch.setattr(experiment, "save_recording", fail)
    out = tmp_path / "run"
    with pytest.raises(OSError, match="full disk"):
        run_experiment(config, out)
    assert out.exists()
    assert not (out / "completion.json").exists()
    with pytest.raises(FileNotFoundError):
        verify_experiment(out)


def test_synthetic_fixture_cli_and_noise_rerun(tmp_path, capsys):
    directory = tmp_path / "synthetic"
    assert main(["make-synthetic-experiment", str(directory)]) == 0
    assert json.loads(capsys.readouterr().out)["configs"] == 5
    assert main(["make-synthetic-experiment", str(directory)]) == 2
    capsys.readouterr()
    config = directory / "raw.json"
    values = json.loads(config.read_text())
    values.update(noise_std_m=.01, dropout_probability=.01)
    config.write_text(json.dumps(values), encoding="utf-8")
    first = run_experiment(config, tmp_path / "run")
    again = run_experiment(tmp_path / "run" / "resolved_config.json", tmp_path / "rerun")
    assert first["display_error"] == again["display_error"]
    assert first["jitter"] == again["jitter"]
    assert main(["verify-experiment", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"


def test_nonoverlapping_latency_windows_report_discarded_tail(tmp_path):
    config = setup_experiment(tmp_path, window_frames=125)
    report = run_experiment(config, tmp_path / "run")
    latency = report["latency"]
    assert len(latency["windows"]) == 3
    assert latency["discarded_tail_frames"] == 5
    assert latency["total_joint_windows"] == 51
    assert latency["windows"][0]["end_ns"] < latency["windows"][1]["start_ns"]
    assert report["display_error"]["possible_joint_pairs"] == 380 * 17


@pytest.mark.parametrize("stress", [
    {"outlier_probability": .01, "outlier_std_m": .2},
    {"outlier_probability": .05, "outlier_magnitude_m": .150},
    {"tracking_gap_period_ms": 1000, "tracking_gap_duration_ms": 250},
    {"gap_period_ms": 1000, "gap_duration_ms": 250},
    {"observation_stride": 6},
    {"noise_std_m": .025, "dropout_probability": .002, "outlier_probability": .01,
     "outlier_std_m": .2, "gap_period_ms": 1000, "gap_duration_ms": 150,
     "observation_stride": 6},
])
def test_extended_stress_is_saved_verified_and_exactly_replayable(tmp_path, stress):
    config = setup_experiment(tmp_path, **stress)
    reference = (tmp_path / "truth.npz").read_bytes()
    report = run_experiment(config, tmp_path / "run")
    replay = run_experiment(tmp_path / "run/resolved_config.json", tmp_path / "replay")
    assert {k: v for k, v in report.items() if k != "runtime"} == {
        k: v for k, v in replay.items() if k != "runtime"
    }
    assert verify_experiment(tmp_path / "run")["status"] == "verified"
    assert (tmp_path / "truth.npz").read_bytes() == reference
    resolved = json.loads((tmp_path / "run/resolved_config.json").read_text())
    for key, value in stress.items():
        assert resolved[key] == value
    used = load_recording(tmp_path / "run/observations_used.npz")
    if "observation_stride" in stress or "gap_period_ms" in stress:
        assert len(used.frames) < 400
    assert used.frames[0].frame_id == 0
