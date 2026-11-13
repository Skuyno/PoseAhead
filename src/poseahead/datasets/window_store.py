"""Checked, one-shard-at-a-time reads of exported predictor windows."""

import json
import re
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.augmentation import validate_noise
from poseahead.datasets.splits import subject_split
from poseahead.datasets.windows import _sampling_settings, _settings


class WindowStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        self.manifest_sha256 = sha256_file(path)
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        if sha256_file(path) != self.manifest_sha256:
            raise ValueError("window manifest changed during loading")
        m = self.manifest
        fields = {"format", "version", "split", "horizon_ms", "history_frames", "step_ns",
                  "stride", "shard_size", "dataset_index", "dataset_index_sha256", "features",
                  "target_interpolation", "sequences", "shards", "windows"}
        if isinstance(m, dict) and m.get("version") in (2, 3):
            fields |= {"history_sampling", "max_history_gap_ns"}
        if isinstance(m, dict) and m.get("version") == 3:
            fields.add("training_noise")
        if (not isinstance(m, dict) or set(m) != fields
                or m["format"] != "poseahead-training-windows"
                or type(m["version"]) is not int or m["version"] not in (1, 2, 3)):
            raise ValueError("invalid training-window manifest")
        if m["version"] in (2, 3):
            _sampling_settings(m["history_sampling"], m["max_history_gap_ns"])
        if m["version"] == 3:
            noise = m["training_noise"]
            if (not isinstance(noise, dict) or set(noise) != {"std_m", "seed", "stream", "targets"}
                    or m["split"] != "train" or noise["targets"] != "clean_reference"
                    or noise["stream"] !=
                    "sha256_seed_sequence_clock_first8_le_PCG64_per_source_frame"):
                raise ValueError("invalid training corruption contract")
            validate_noise(noise["std_m"], noise["seed"])
        _settings(m["history_frames"], m["horizon_ms"], m["step_ns"], m["stride"])
        if m["split"] not in ("train", "validation", "test"):
            raise ValueError("invalid window split")
        if (m["features"] != "canonical17_root_relative_positions_m_no_normalization"
                or m["target_interpolation"] != "linear_reference_only_max_gap_step_ns"):
            raise ValueError("unsupported window feature/interpolation convention")
        if type(m["shard_size"]) is not int or not 1 <= m["shard_size"] <= 4096:
            raise ValueError("invalid shard size")
        if type(m["windows"]) is not int or m["windows"] < 0:
            raise ValueError("invalid window count")
        if not isinstance(m["sequences"], list) or not m["sequences"]:
            raise ValueError("window manifest requires source sequences")
        if not isinstance(m["shards"], list):
            raise ValueError("invalid shard list")
        identities = set()
        for entry in m["sequences"]:
            if not isinstance(entry, dict):
                raise ValueError("invalid source entry")
            subject = entry.get("subject")
            if (not isinstance(subject, str) or subject_split(subject) != m["split"]
                    or entry.get("split") != m["split"]):
                raise ValueError("source subject violates window split")
            identity = (entry.get("sequence_id"), entry.get("clock_domain"))
            if any(not isinstance(s, str) or not s for s in identity) or identity in identities:
                raise ValueError("invalid or duplicate sequence identity")
            identities.add(identity)
            if any(type(entry.get(k)) is not int or not 0 <= entry[k] < 2**64
                   for k in ("start_ns", "end_ns")) or entry["end_ns"] < entry["start_ns"]:
                raise ValueError("invalid source time bounds")
        names, total = set(), 0
        for shard in m["shards"]:
            if not isinstance(shard, dict) or set(shard) != {
                "file", "sha256", "windows", "sequence_index",
            }:
                raise ValueError("invalid shard entry")
            name = shard["file"]
            if (not isinstance(name, str) or not re.fullmatch(r"shard-\d{6}\.npz", name)
                    or name in names):
                raise ValueError("unsafe or duplicate shard filename")
            if (type(shard["windows"]) is not int or not 1 <= shard["windows"] <= m["shard_size"]
                    or type(shard["sequence_index"]) is not int
                    or not 0 <= shard["sequence_index"] < len(m["sequences"])):
                raise ValueError("invalid shard dimensions/sequence")
            names.add(name)
            total += shard["windows"]
        if total != m["windows"]:
            raise ValueError("shard window counts disagree with manifest")

    def iter_shards(self):
        if sha256_file(self.directory / "manifest.json") != self.manifest_sha256:
            raise ValueError("window manifest changed after opening")
        m = self.manifest
        previous_by_sequence = {}
        for shard in m["shards"]:
            path = self.directory / shard["file"]
            if path.is_symlink() or sha256_file(path) != shard["sha256"]:
                raise ValueError("window shard checksum mismatch")
            count, history = shard["windows"], m["history_frames"]
            schema = {
                "history_positions_m": (np.float32, (count, history, 17, 3)),
                "history_timestamps_ns": (np.uint64, (count, history)),
                "target_positions_m": (np.float32, (count, 17, 3)),
                "target_timestamp_ns": (np.uint64, (count,)),
                "anchor_frame_id": (np.uint64, (count,)),
            }
            try:
                loaded = np.load(path, allow_pickle=False)
                if not isinstance(loaded, np.lib.npyio.NpzFile):
                    raise ValueError("window shard must be an NPZ archive")
                with loaded as archive:
                    if set(archive.files) != set(schema) or len(archive.files) != len(schema):
                        raise ValueError("invalid window shard fields")
                    arrays = {name: archive[name] for name in schema}
            except (BadZipFile, EOFError, TypeError) as error:
                raise ValueError("invalid window archive") from error
            if sha256_file(path) != shard["sha256"]:
                raise ValueError("window shard changed during loading")
            for name, (dtype, shape) in schema.items():
                array = arrays[name]
                if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"invalid window array: {name}")
            for name in ("history_positions_m", "target_positions_m"):
                if not np.allclose(arrays[name][..., 0, :], 0, atol=1e-6):
                    raise ValueError("window pelvis must remain zero")
            source = m["sequences"][shard["sequence_index"]]
            for times, target, frame_id in zip(arrays["history_timestamps_ns"],
                                               arrays["target_timestamp_ns"],
                                               arrays["anchor_frame_id"], strict=True):
                times = [int(t) for t in times]
                anchor = times[-1]
                if times != [anchor - (history - 1 - i) * m["step_ns"] for i in range(history)]:
                    raise ValueError("noncausal or irregular window history")
                if int(target) != anchor + m["horizon_ms"] * 1_000_000:
                    raise ValueError("window target horizon mismatch")
                if times[0] < source["start_ns"] or int(target) > source["end_ns"]:
                    raise ValueError("window crosses source clip bounds")
                previous = previous_by_sequence.get(shard["sequence_index"])
                if previous is not None and (anchor <= previous[0] or int(frame_id) <= previous[1]):
                    raise ValueError("duplicate or backwards windows within source clip")
                previous_by_sequence[shard["sequence_index"]] = (anchor, int(frame_id))
            yield arrays

    def verify(self):
        count = sum(len(shard["target_timestamp_ns"]) for shard in self.iter_shards())
        return {"status": "verified", "windows": count, "split": self.manifest["split"],
                "horizon_ms": self.manifest["horizon_ms"], "manifest_sha256": self.manifest_sha256}
