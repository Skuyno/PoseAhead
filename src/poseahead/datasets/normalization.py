"""Train-only streaming normalization, shared unchanged by history and targets."""

import json
from pathlib import Path

import numpy as np

from poseahead.datasets.window_store import WindowStore


def fit_normalizer(directory, output, *, minimum_scale_m=.001):
    if type(minimum_scale_m) not in (int, float) or not np.isfinite(minimum_scale_m):
        raise ValueError("normalization scale floor must be finite numeric")
    if minimum_scale_m <= 0:
        raise ValueError("normalization scale floor must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("normalizer already exists")
    store = WindowStore(directory)
    if store.manifest["split"] != "train":
        raise ValueError("normalizer may only be fitted on train windows")
    count = 0
    mean, m2 = np.zeros((17, 3)), np.zeros((17, 3))
    for shard in store.iter_shards():
        # Only histories contribute. Overlapping appearances receive equal weight;
        # this is window-weighted, not unique-frame or subject-weighted statistics.
        samples = shard["history_positions_m"].reshape(-1, 17, 3).astype(np.float64)
        batch_count = len(samples)
        batch_mean = samples.mean(axis=0)
        batch_m2 = ((samples - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - mean
        total = count + batch_count
        m2 += batch_m2 + delta**2 * count * batch_count / total
        mean += delta * batch_count / total
        count = total
    if not count:
        raise ValueError("cannot fit normalization on an empty training export")
    scale = np.maximum(np.sqrt(m2 / count), minimum_scale_m)
    mean[0], scale[0] = 0, 1  # pelvis must stay exactly zero, including inverse transform
    report = {
        "format": "poseahead-normalizer", "version": 1, "fit_split": "train",
        "source_manifest_sha256": store.manifest_sha256,
        "dataset_index_sha256": store.manifest["dataset_index_sha256"],
        "history_frames": store.manifest["history_frames"], "step_ns": store.manifest["step_ns"],
        "horizon_ms": store.manifest["horizon_ms"], "sample_count": count,
        "weighting": "each_history_occurrence_equal_targets_excluded_population_variance",
        "minimum_scale_m": minimum_scale_m, "mean_m": mean.tolist(), "scale_m": scale.tolist(),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return {"fit_split": "train", "history_samples": count,
            "source_manifest_sha256": store.manifest_sha256}


class PoseNormalizer:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text(encoding="utf-8"))
        c = self.config
        if (not isinstance(c, dict) or c.get("format") != "poseahead-normalizer"
                or type(c.get("version")) is not int or c["version"] != 1
                or c.get("fit_split") != "train"):
            raise ValueError("invalid train normalizer")
        self.mean = np.asarray(c.get("mean_m"), dtype=np.float64)
        self.scale = np.asarray(c.get("scale_m"), dtype=np.float64)
        if (self.mean.shape != (17, 3) or self.scale.shape != (17, 3)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or not (self.scale > 0).all() or np.any(self.mean[0] != 0)
                or np.any(self.scale[0] != 1)):
            raise ValueError("invalid normalization mean/scale")
        self.mean = np.frombuffer(self.mean.tobytes(), dtype=np.float64).reshape(17, 3)
        self.scale = np.frombuffer(self.scale.tobytes(), dtype=np.float64).reshape(17, 3)

    def check_compatible(self, store):
        for field in ("dataset_index_sha256", "history_frames", "step_ns", "horizon_ms"):
            if field not in self.config or self.config[field] != store.manifest[field]:
                raise ValueError(f"normalizer/window mismatch: {field}")

    def transform(self, positions, *, inverse=False):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape[-2:] != (17, 3) or not np.isfinite(positions).all():
            raise ValueError("normalizer requires finite (...,17,3) positions")
        result = (positions * self.scale + self.mean if inverse
                  else (positions - self.mean) / self.scale)
        return result
