"""Numeric-only checkpoint loading; never deserialize arbitrary Python objects."""

import json
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.normalization import PoseNormalizer


class FixedHorizonPredictor:
    def __init__(self, directory):
        import torch

        from poseahead.prediction.mlp import ResidualMLP

        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("format") != "poseahead-residual-mlp"
                or type(manifest.get("version")) is not int
                or manifest["version"] not in (1, 2, 3)):
            raise ValueError("invalid predictor checkpoint manifest")
        features = manifest.get("input_features", "positions")
        if (features not in ("positions", "relative_last")
                or (manifest["version"] == 1 and features != "positions")
                or (manifest["version"] >= 2 and "input_features" not in manifest)):
            raise ValueError("invalid checkpoint input_features")
        baseline = manifest.get("output_baseline", "hold")
        if (baseline not in ("hold", "constant_velocity")
                or (manifest["version"] < 3 and baseline != "hold")
                or (manifest["version"] == 3 and "output_baseline" not in manifest)):
            raise ValueError("invalid checkpoint output_baseline")
        horizon = manifest.get("horizon_ms")
        if type(horizon) is not int or horizon not in (33, 66, 100, 150):
            raise ValueError("unsupported predictor horizon")
        if type(manifest.get("step_ns")) is not int or manifest["step_ns"] <= 0:
            raise ValueError("invalid predictor sampling interval")
        for name, key in (("model.npz", "model_sha256"), ("normalizer.json", "normalizer_sha256")):
            if sha256_file(directory / name) != manifest.get(key):
                raise ValueError(f"checkpoint checksum mismatch: {name}")
        normalizer = PoseNormalizer(directory / "normalizer.json")
        for field in ("history_frames", "horizon_ms", "step_ns"):
            if normalizer.config.get(field) != manifest.get(field):
                raise ValueError("checkpoint/normalizer contract mismatch")
        if normalizer.config.get("source_manifest_sha256") != manifest.get("train_manifest_sha256"):
            raise ValueError("checkpoint/normalizer training provenance mismatch")
        with torch.random.fork_rng(devices=[]):
            model = ResidualMLP(manifest.get("history_frames"), manifest.get("hidden_size"),
                                input_features=features, output_baseline=baseline,
                                horizon_steps=(horizon * 1_000_000 / manifest["step_ns"]
                                               if baseline == "constant_velocity" else None))
        loaded = np.load(directory / "model.npz", allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("predictor weights must be NPZ")
        with loaded as archive:
            expected = model.state_dict()
            if len(archive.files) != len(expected) or set(archive.files) != set(expected):
                raise ValueError("checkpoint parameter names mismatch")
            state = {}
            for name, tensor in expected.items():
                array = archive[name]
                if (array.dtype != np.float32 or array.shape != tuple(tensor.shape)
                        or not np.isfinite(array).all()):
                    raise ValueError(f"invalid model parameter: {name}")
                state[name] = torch.from_numpy(array.copy())
        for name, key in (("model.npz", "model_sha256"), ("normalizer.json", "normalizer_sha256")):
            if sha256_file(directory / name) != manifest[key]:
                raise ValueError("checkpoint changed during loading")
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model, self.normalizer = model, normalizer
        self.history_frames, self.horizon_ms = manifest["history_frames"], horizon
        self.step_ns = manifest["step_ns"]

    def predict(self, positions, timestamps_ns):
        import torch

        positions = np.asarray(positions)
        if positions.shape != (self.history_frames, 17, 3):
            raise ValueError("predictor history shape mismatch")
        times = tuple(timestamps_ns)
        if (len(times) != self.history_frames
                or any(type(t) is not int or not 0 <= t < 2**64 for t in times)
                or times != tuple(times[-1] - (self.history_frames - 1 - i) * self.step_ns
                                  for i in range(self.history_frames))):
            raise ValueError("predictor requires its trained uniform time grid")
        if times[-1] + self.horizon_ms * 1_000_000 >= 2**64:
            raise ValueError("prediction target exceeds uint64")
        if not np.allclose(positions[:, 0, :], 0, atol=1e-6):
            raise ValueError("predictor requires root-relative history")
        inputs = self.normalizer.transform(positions).astype(np.float32)
        with torch.no_grad():
            prediction = self.model(torch.from_numpy(inputs[None])).numpy()[0]
        result = self.normalizer.transform(prediction, inverse=True).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("predictor produced nonfinite output")
        return result
