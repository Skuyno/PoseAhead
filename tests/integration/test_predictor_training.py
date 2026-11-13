import json
from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseSequence
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.index import freeze_dataset
from poseahead.datasets.normalization import fit_normalizer
from poseahead.datasets.windows import export_training_windows
from poseahead.evaluation.experiment import run_experiment
from poseahead.prediction.checkpoint import FixedHorizonPredictor
from poseahead.prediction.training import train_predictor
from poseahead.recording import save_recording

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("features,order", [("positions", "global_windows"),
                                           ("relative_last", "within_shards"),
                                           ("relative_last", "global_windows")])
def test_feature_order_variants_reload_and_repeat(training_data, tmp_path, features, order):
    config = json.loads(training_data.read_text())
    config.update(input_features=features, batch_order=order)
    training_data.write_text(json.dumps(config))
    rng_before = torch.random.get_rng_state().clone()
    first = train_predictor(training_data, tmp_path / "variant")
    repeated = train_predictor(training_data, tmp_path / "repeat_variant")
    assert first == repeated
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    predictor = FixedHorizonPredictor(tmp_path / "variant")
    assert predictor.model.input_features == features
    assert first["best_epoch"] > 0
    positions = np.zeros((3, 17, 3), np.float32)
    assert np.isfinite(predictor.predict(positions, (0, 20_000_000, 40_000_000))).all()
    if features == "relative_last":
        path = tmp_path / "variant" / "manifest.json"
        manifest = json.loads(path.read_text())
        assert manifest["version"] == 2
        manifest["version"] = 1
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="input_features"):
            FixedHorizonPredictor(tmp_path / "variant")


def test_relative_features_ignore_constant_joint_offsets():
    from poseahead.prediction.mlp import ResidualMLP
    with torch.random.fork_rng(devices=[]):
        model = ResidualMLP(3, 8, input_features="relative_last")
        torch.nn.init.normal_(model.network[-1].weight)
        history = torch.randn(2, 3, 17, 3)
        history[:, :, 0] = 0
        offset = torch.randn(2, 1, 17, 3)
        offset[:, :, 0] = 0
        torch.testing.assert_close(model(history + offset), model(history) + offset[:, 0])


def test_velocity_residual_starts_at_physical_velocity_with_normalization():
    from poseahead.prediction.mlp import ResidualMLP

    with torch.random.fork_rng(devices=[]):
        model = ResidualMLP(3, 8, input_features="relative_last",
                            output_baseline="constant_velocity", horizon_steps=1.65)
        history = torch.randn(2, 3, 17, 3)
        history[:, :, 0] = 0
        mean, scale = torch.randn(17, 3), torch.rand(17, 3) + .1
        mean[0], scale[0] = 0, 1
        expected = history[:, -1] + (history[:, -1] - history[:, -2]) * 1.65
        actual = model((history - mean) / scale) * scale + mean
        torch.testing.assert_close(actual, expected)
        # The baseline is differentiable and the zero correction can learn.
        model((history - mean) / scale).square().mean().backward()
        assert model.network[-1].weight.grad.abs().sum() > 0
        torch.nn.init.normal_(model.network[-1].weight)
        offset = torch.randn(2, 1, 17, 3)
        offset[:, :, 0] = 0
        torch.testing.assert_close(model(history + offset), model(history) + offset[:, 0])


@pytest.mark.parametrize("frames,baseline,steps", [
    (1, "constant_velocity", 5), (3, "constant_velocity", None),
    (3, "constant_velocity", 0), (3, "constant_velocity", -1),
    (3, "constant_velocity", True), (3, "constant_velocity", float("nan")),
    (3, "constant_velocity", float("inf")), (3, "hold", 5), (3, "unknown", None),
])
def test_invalid_velocity_contract(frames, baseline, steps):
    from poseahead.prediction.mlp import ResidualMLP

    with pytest.raises(ValueError):
        ResidualMLP(frames, 8, output_baseline=baseline, horizon_steps=steps)


def test_velocity_checkpoint_roundtrip_and_downgrade_rejected(training_data, tmp_path):
    config = json.loads(training_data.read_text())
    config.update(input_features="relative_last", output_baseline="constant_velocity",
                  batch_order="global_windows")
    training_data.write_text(json.dumps(config))
    before = torch.random.get_rng_state().clone()
    report = train_predictor(training_data, tmp_path / "velocity")
    repeat = train_predictor(training_data, tmp_path / "velocity_repeat")
    assert report == repeat
    predictor = FixedHorizonPredictor(tmp_path / "velocity")
    assert torch.equal(before, torch.random.get_rng_state())
    assert predictor.model.output_baseline == "constant_velocity"
    assert predictor.model.horizon_steps == 1.65
    manifest_path = tmp_path / "velocity" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 3
    assert "epoch0_constant_velocity_eligible" in manifest["selection"]
    # Synthetic motion is linear: epoch zero already solves this fixture.
    assert manifest["epochs"][0]["validation_mpjpe_mm"] < 1e-4
    positions = np.zeros((3, 17, 3), np.float32)
    positions[:, 1:, 0] = np.array([.1, .12, .14])[:, None]
    expected = np.zeros((17, 3), np.float32)
    expected[1:, 0] = .173
    np.testing.assert_allclose(predictor.predict(positions, (0, 20_000_000, 40_000_000)),
                               expected, atol=1e-6)
    for version in (1, 2):
        manifest_path.write_text(json.dumps({**manifest, "version": version,
                                             "input_features": "positions"}))
        with pytest.raises(ValueError, match="output_baseline"):
            FixedHorizonPredictor(tmp_path / "velocity")
    manifest.pop("output_baseline")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="output_baseline"):
        FixedHorizonPredictor(tmp_path / "velocity")


@pytest.fixture
def training_data(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    paths = []
    for subject, split in (("S1", "train"), ("S8", "validation"), ("S9", "test")):
        points = np.zeros((35, 32, 3))
        points[:, 1:, 0] = np.arange(35)[:, None] * 20
        sequence = camera_poses_to_sequence(points, sequence_id=subject, camera_id="cam")
        frames = tuple(replace(f, metadata={**f.metadata, "subject": subject, "split": split})
                       for f in sequence.frames)
        path = tmp_path / f"{subject}.npz"
        save_recording(path, PoseSequence(sequence.sequence_id, sequence.clock_domain, frames))
        paths.append(path)
    index = tmp_path / "index.json"
    freeze_dataset(paths, index)
    for split in ("train", "validation", "test"):
        export_training_windows(index, tmp_path / split, split=split, horizon_ms=33,
                                history_frames=3, shard_size=16)
    fit_normalizer(tmp_path / "train", tmp_path / "normalizer.json")
    config = {"version": 1, "train_windows": "train", "validation_windows": "validation",
              "normalizer": "normalizer.json", "seed": 42, "epochs": 8,
              "batch_size": 16, "hidden_size": 8, "learning_rate": .01}
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    try:
        yield config_path
    finally:
        torch.set_num_threads(old_threads)


def test_cpu_training_learns_synthetic_motion_and_checkpoint_roundtrip(training_data, tmp_path):
    rng_before = torch.random.get_rng_state().clone()
    report = train_predictor(training_data, tmp_path / "model")
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert report["best_epoch"] > 0
    manifest = json.loads((tmp_path / "model" / "manifest.json").read_text())
    assert report["validation_mpjpe_mm"] < manifest["epochs"][0]["validation_mpjpe_mm"]
    assert report["test_data_used"] is False
    a, b = FixedHorizonPredictor(tmp_path / "model"), FixedHorizonPredictor(tmp_path / "model")
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    history = np.zeros((3, 17, 3), np.float32)
    history[:, 1:, 0] = np.array([.1, .12, .14])[:, None]
    times = (100_000_000, 120_000_000, 140_000_000)
    output = a.predict(history, times)
    np.testing.assert_array_equal(output, b.predict(history, times))
    np.testing.assert_array_equal(output[0], [0, 0, 0])
    assert output.shape == (17, 3)
    assert (np.linalg.norm(output[1:] - [.173, 0, 0])
            < np.linalg.norm(history[-1, 1:] - [.173, 0, 0]))
    repeated = train_predictor(training_data, tmp_path / "repeat")
    assert repeated == report
    with np.load(tmp_path / "model" / "model.npz", allow_pickle=False) as first:
        with np.load(tmp_path / "repeat" / "model.npz", allow_pickle=False) as second:
            for name in first.files:
                np.testing.assert_array_equal(first[name], second[name])
    with pytest.raises(ValueError, match="grid"):
        a.predict(history, (100_000_000, 121_000_000, 140_000_000))
    model_path = tmp_path / "model" / "model.npz"
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        FixedHorizonPredictor(tmp_path / "model")


@pytest.mark.parametrize("field", ["train_windows", "validation_windows"])
def test_test_partition_excluded_from_training_and_selection(training_data, tmp_path, field):
    config = json.loads(training_data.read_text())
    config[field] = "test"
    training_data.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="test is never"):
        train_predictor(training_data, tmp_path / "model")
    assert not (tmp_path / "model").exists()


@pytest.mark.parametrize("baseline", ["hold", "constant_velocity"])
def test_checkpoint_shared_experiment_and_provenance(training_data, tmp_path, baseline):
    config = json.loads(training_data.read_text())
    config["output_baseline"] = baseline
    training_data.write_text(json.dumps(config))
    train_predictor(training_data, tmp_path / "model")
    config = {"version": 1, "run_id": "learned-synthetic", "observations": "S8.npz",
              "reference": "S8.npz", "processor": {"method": "learned",
                                                       "parameters": {"checkpoint": "model"}},
              "seed": 42, "delay_ms": 33, "start_ns": 0, "end_ns": 680_000_000,
              "step_ns": 20_000_000, "window_frames": 35, "max_lag_ms": 100}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = run_experiment(path, tmp_path / "run")
    assert report["learned_prediction"]["observation_output_counts"] == {
        "warmup": 2, "predicted": 31,
    }
    resolved = tmp_path / "run" / "resolved_config.json"
    parameters = json.loads(resolved.read_text())["processor"]["parameters"]
    assert parameters["checkpoint_sha256"]
    again = run_experiment(resolved, tmp_path / "rerun")
    assert again["display_error"] == report["display_error"]
    assert again["learned_prediction"] == report["learned_prediction"]
    model_manifest = tmp_path / "model" / "manifest.json"
    model_manifest.write_bytes(model_manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        run_experiment(resolved, tmp_path / "changed_model")
    assert not (tmp_path / "changed_model").exists()
