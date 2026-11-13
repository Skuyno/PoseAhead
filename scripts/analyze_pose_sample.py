"""Summarize a directory of serialized canonical PAFR v1 frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poseahead.core import JOINT_PARENTS, PoseSequence
from poseahead.core.serialization import decode_pose_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.sample_dir.glob("*.pafr"))
    if not paths:
        parser.error(f"no .pafr frames found in {args.sample_dir}")
    frames = [decode_pose_frame(path.read_bytes()) for path in paths]
    sequence = PoseSequence(str(args.sample_dir), "recorded_producer_monotonic", tuple(frames))
    if any(frame.root_position is None for frame in frames):
        raise RuntimeError("sample contains frames without root positions")
    roots = np.stack([frame.root_position for frame in frames])
    positions = np.stack([frame.joint_positions for frame in frames])
    bone_lengths = np.stack(
        [
            np.linalg.norm(positions[:, joint] - positions[:, parent], axis=1)
            for joint, parent in enumerate(JOINT_PARENTS)
            if parent >= 0
        ],
        axis=1,
    )
    root_steps = np.linalg.norm(np.diff(roots, axis=0), axis=1)
    report = {
        "sample_dir": str(args.sample_dir),
        "frames": len(frames),
        "event_duration_ns": sequence.duration_ns,
        "all_frames_valid": all(frame.is_valid for frame in frames),
        "first_frame_id": frames[0].frame_id,
        "last_frame_id": frames[-1].frame_id,
        "root_min_m": roots.min(axis=0).tolist(),
        "root_max_m": roots.max(axis=0).tolist(),
        "root_step_p95_m": float(np.percentile(root_steps, 95)) if len(root_steps) else 0.0,
        "median_bone_length_m": float(np.median(bone_lengths)),
        "mean_per_bone_temporal_cv": float(
            np.mean(
                np.std(bone_lengths, axis=0)
                / np.maximum(np.mean(bone_lengths, axis=0), 1e-9)
            )
        ),
        "interpretation": (
            "Internal continuity summary only; it is not ground-truth metric accuracy."
        ),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
