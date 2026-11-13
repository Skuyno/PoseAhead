"""Explicit, pickle-free import of prepared Human3.6M numeric exports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from poseahead.core import PoseSequence
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.splits import SUBJECT_SPLITS


def load_human36m(manifest_path: str | Path) -> PoseSequence:
    """Load a JSON manifest plus NPY positions. No implicit camera conventions."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"format", "version", "subject", "action", "take", "camera_id",
                "positions_file", "coordinate_space", "units", "fps_numerator"}
    optional = {"fps_denominator", "first_frame_index", "rotation_world_to_camera",
                "translation_world_to_camera_mm"}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError("incomplete Human3.6M manifest")
    if manifest.keys() - required - optional:
        raise ValueError("unknown Human3.6M manifest fields")
    if manifest["format"] != "poseahead-human36m" or (
        type(manifest["version"]) is not int or manifest["version"] != 1
    ):
        raise ValueError("unsupported Human3.6M manifest")
    for field in ("subject", "action", "take", "camera_id", "positions_file"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if manifest["subject"] not in SUBJECT_SPLITS:
        raise ValueError("subject is outside the fixed Human3.6M split")
    if manifest["units"] != "mm":
        raise ValueError("only explicit millimetre exports are supported")
    if manifest["coordinate_space"] not in ("camera", "world"):
        raise ValueError("coordinate_space must be camera or world")
    positions_path = path.parent / manifest["positions_file"]
    if positions_path.suffix.lower() != ".npy":
        raise ValueError("positions_file must be a numeric NPY file")
    positions = np.load(positions_path, allow_pickle=False)
    if not isinstance(positions, np.ndarray):
        positions.close()
        raise ValueError("positions_file must contain a numeric NPY array, not an archive")
    if positions.dtype.kind not in "fiu":
        raise ValueError("positions must be a real numeric array")
    if positions.ndim != 3 or positions.shape[1:] != (32, 3):
        raise ValueError("positions must have shape (N, 32, 3)")
    positions = positions.astype(np.float64)
    extrinsics = {"rotation_world_to_camera", "translation_world_to_camera_mm"}
    if manifest["coordinate_space"] == "world":
        if not extrinsics <= manifest.keys():
            raise ValueError("world coordinates require explicit world-to-camera R and t")
        rotation = np.asarray(manifest["rotation_world_to_camera"], dtype=np.float64)
        translation = np.asarray(manifest["translation_world_to_camera_mm"], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("camera extrinsics require R(3,3) and t(3)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("camera extrinsics must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0) or (
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError("camera rotation must be a proper orthonormal matrix")
        positions = positions @ rotation.T + translation
    elif extrinsics & manifest.keys():
        raise ValueError("camera-space positions must not also declare extrinsics")
    sequence_id = "/".join(manifest[field] for field in ("subject", "action", "take"))
    sequence = camera_poses_to_sequence(
        positions, sequence_id=sequence_id, camera_id=manifest["camera_id"],
        fps_numerator=manifest["fps_numerator"],
        fps_denominator=manifest.get("fps_denominator", 1),
        first_frame_index=manifest.get("first_frame_index", 0),
    )
    frames = tuple(replace(frame, metadata={
        **frame.metadata, "subject": manifest["subject"], "action": manifest["action"],
        "take": manifest["take"], "split": SUBJECT_SPLITS[manifest["subject"]],
        "input_coordinate_space": manifest["coordinate_space"],
    }) for frame in sequence.frames)
    return PoseSequence(sequence.sequence_id, sequence.clock_domain, frames)
