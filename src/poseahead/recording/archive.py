"""Self-contained NPZ recordings with numeric arrays and UTF-8 JSON, never pickle."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from poseahead.core import PoseFrame, PoseSequence
from poseahead.core.metadata import metadata_to_json

_ARRAYS = {
    "manifest", "ids", "source_ns", "produced_ns", "positions", "confidence", "validity",
    "optional_present", "roots", "root_rotations", "joint_rotations",
}


def save_recording(path: str | Path, sequence: PoseSequence) -> None:
    """Write one complete sequence. Existing recordings are never overwritten.

    A completed temporary file is published without replacing an existing path.
    The caller controls the parent
    directory; raw observations and derived runs should have distinct filenames.
    """
    frames = sequence.frames
    manifest = {
        "format": "poseahead-recording", "version": 1,
        "sequence_id": sequence.sequence_id, "clock_domain": sequence.clock_domain,
        "frame_count": len(frames),
        "metadata": [metadata_to_json(frame.metadata) for frame in frames],
    }
    encoded = json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    present = np.array([
        [f.root_position is not None, f.root_rotation is not None,
         f.joint_rotations is not None] for f in frames
    ], dtype=bool)
    arrays = {
        "manifest": np.frombuffer(encoded, dtype=np.uint8),
        "ids": np.array([f.frame_id for f in frames], dtype=np.uint64),
        "source_ns": np.array([f.source_timestamp_ns for f in frames], dtype=np.uint64),
        "produced_ns": np.array([f.produced_timestamp_ns for f in frames], dtype=np.uint64),
        "positions": np.stack([f.joint_positions for f in frames]),
        "confidence": np.stack([f.joint_confidence for f in frames]),
        "validity": np.stack([f.joint_validity for f in frames]),
        "optional_present": present,
        "roots": np.stack([
            f.root_position if f.root_position is not None else np.zeros(3, np.float32)
            for f in frames
        ]),
        "root_rotations": np.stack([
            f.root_rotation if f.root_rotation is not None else np.zeros(4, np.float32)
            for f in frames
        ]),
        "joint_rotations": np.stack([
            f.joint_rotations if f.joint_rotations is not None else np.zeros((17, 4), np.float32)
            for f in frames
        ]),
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"recording already exists: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".poseahead-", delete=False
        ) as out:
            temporary = Path(out.name)
            np.savez_compressed(out, **arrays)
            out.flush()
            os.fsync(out.fileno())
        # Hard-link creation is atomic and fails if another writer won this name.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def load_recording(path: str | Path) -> PoseSequence:
    """Decode a v1 recording and validate array schema, poses and clock ordering."""
    try:
        return _load_recording(path)
    except (BadZipFile, EOFError) as error:
        raise ValueError(f"damaged recording archive: {error}") from error


def _load_recording(path: str | Path) -> PoseSequence:
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("recording must be an NPZ archive")
    with loaded as archive:
        if len(archive.files) != len(_ARRAYS) or set(archive.files) != _ARRAYS:
            raise ValueError("unexpected recording fields")
        encoded = archive["manifest"]
        if encoded.dtype != np.uint8 or encoded.ndim != 1:
            raise ValueError("manifest must be a UTF-8 byte vector")
        manifest = json.loads(encoded.tobytes().decode("utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format") != "poseahead-recording":
            raise ValueError("invalid recording manifest")
        if type(manifest.get("version")) is not int or manifest["version"] != 1:
            raise ValueError("unsupported recording version")
        count = manifest.get("frame_count")
        if type(count) is not int or count < 1:
            raise ValueError("frame_count must be a positive integer")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, list) or len(metadata) != count:
            raise ValueError("metadata count does not match frame_count")
        schema = {
            "ids": (np.uint64, (count,)), "source_ns": (np.uint64, (count,)),
            "produced_ns": (np.uint64, (count,)),
            "positions": (np.float32, (count, 17, 3)),
            "confidence": (np.float32, (count, 17)), "validity": (np.bool_, (count, 17)),
            "optional_present": (np.bool_, (count, 3)), "roots": (np.float32, (count, 3)),
            "root_rotations": (np.float32, (count, 4)),
            "joint_rotations": (np.float32, (count, 17, 4)),
        }
        arrays = {}
        for name, (dtype, shape) in schema.items():
            array = archive[name]
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(f"invalid recording array: {name}")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"nonfinite recording array: {name}")
            arrays[name] = array
    frames = []
    for i in range(count):
        present = arrays["optional_present"][i]
        frames.append(PoseFrame(
            frame_id=int(arrays["ids"][i]), source_timestamp_ns=int(arrays["source_ns"][i]),
            produced_timestamp_ns=int(arrays["produced_ns"][i]),
            joint_positions=arrays["positions"][i], joint_confidence=arrays["confidence"][i],
            joint_validity=arrays["validity"][i],
            root_position=arrays["roots"][i] if present[0] else None,
            root_rotation=arrays["root_rotations"][i] if present[1] else None,
            joint_rotations=arrays["joint_rotations"][i] if present[2] else None,
            metadata=metadata[i],
        ))
    return PoseSequence(manifest.get("sequence_id"), manifest.get("clock_domain"), tuple(frames))
