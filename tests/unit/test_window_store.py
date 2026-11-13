import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseSequence
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.index import freeze_dataset
from poseahead.datasets.normalization import PoseNormalizer, fit_normalizer
from poseahead.datasets.window_store import WindowStore
from poseahead.datasets.windows import export_training_windows
from poseahead.recording import save_recording


def prepare(tmp_path):
    paths = []
    for subject, split in (("S1", "train"), ("S8", "validation"), ("S9", "test")):
        points = np.zeros((20, 32, 3))
        points[:, 1:, 0] = np.arange(20)[:, None] * (20 if split == "train" else 200)
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
                                history_frames=3, shard_size=4)
    return tmp_path / "train"


def test_streaming_normalization_matches_numpy_and_roundtrip(tmp_path):
    directory = prepare(tmp_path)
    store = WindowStore(directory)
    assert store.verify()["windows"] == 16
    output = tmp_path / "normalizer.json"
    fit_normalizer(directory, output)
    normalizer = PoseNormalizer(output)
    samples = np.concatenate([s["history_positions_m"].reshape(-1, 17, 3)
                              for s in store.iter_shards()]).astype(np.float64)
    np.testing.assert_allclose(normalizer.mean, samples.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(normalizer.scale[1:],
                               np.maximum(samples.std(axis=0)[1:], .001), atol=1e-12)
    np.testing.assert_allclose(normalizer.transform(normalizer.transform(samples), inverse=True),
                               samples, atol=1e-12)
    for split in ("validation", "test"):
        normalizer.check_compatible(WindowStore(tmp_path / split))
        with pytest.raises(ValueError, match="only.*train"):
            fit_normalizer(tmp_path / split, tmp_path / f"{split}.json")
        assert not (tmp_path / f"{split}.json").exists()
    with pytest.raises(FileExistsError):
        fit_normalizer(directory, output)


@pytest.mark.parametrize("corruption", ["dtype", "target", "history", "pelvis", "nan"])
def test_shard_semantics_rejected_even_if_hash_recomputed(tmp_path, corruption):
    directory = prepare(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard_path = directory / manifest["shards"][0]["file"]
    with np.load(shard_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if corruption == "dtype":
        arrays["anchor_frame_id"] = arrays["anchor_frame_id"].astype(np.float64)
    elif corruption == "target":
        arrays["target_timestamp_ns"][0] += np.uint64(1)
    elif corruption == "history":
        arrays["history_timestamps_ns"][0, 0] += np.uint64(1)
    elif corruption == "pelvis":
        arrays["target_positions_m"][0, 0, 0] = 1
    else:
        arrays["history_positions_m"][0, 0, 1, 0] = np.nan
    np.savez(shard_path, **arrays)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        WindowStore(directory).verify()


def test_checksum_and_manifest_change_detected(tmp_path):
    directory = prepare(tmp_path)
    store = WindowStore(directory)
    path = directory / store.manifest["shards"][0]["file"]
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="checksum"):
        store.verify()
    manifest = directory / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest changed"):
        store.verify()


@pytest.mark.parametrize("corruption", ["split", "path", "count", "sequence", "horizon"])
def test_bad_manifest(tmp_path, corruption):
    directory = prepare(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "split":
        manifest["split"] = "test"
    elif corruption == "path":
        manifest["shards"][0]["file"] = "../outside.npz"
    elif corruption == "count":
        manifest["windows"] += 1
    elif corruption == "sequence":
        manifest["shards"][0]["sequence_index"] = 100
    else:
        manifest["horizon_ms"] = 50
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        WindowStore(directory)
