"""Freeze SPEC E2 acceleration classes on training ground truth, then label validation."""

import argparse
import json
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.index import verify_dataset
from poseahead.datasets.splits import sequence_split
from poseahead.recording import load_recording

_PARAMETERS = {
    "grid_step_ns": 25_000_000, "window_frames": 40,
    "filter": "centered_gaussian_sigma25ms_support3sigma_edge_padding",
    "derivative": "two_numpy_gradient_edge_order2_on_40hz_grid",
    "score": "p95_of_all_time_nonpelvis_joint_acceleration_magnitudes_in_window",
    "quantiles": [0.30, 0.70], "quantile_method": "linear",
    "invalid_policy": "unknown_if_any_nonpelvis_filter_derivative_support_invalid",
    "equal_threshold_tie": "smooth_precedes_high_dynamic",
}


def window_scores(positions, validity, timestamps_ns, *, window_offset=0):
    """Ground-truth-only labels, never causal predictor inputs or method error labels."""
    positions = np.asarray(positions, dtype=np.float64)
    validity = np.asarray(validity)
    times = tuple(timestamps_ns)
    if (positions.shape != (len(times), 17, 3) or validity.shape != (len(times), 17)
            or not np.isfinite(positions).all() or validity.dtype.kind != "b"
            or any(type(t) is not int or not 0 <= t < 2**64 for t in times)
            or any(b - a != 25_000_000 for a, b in zip(times, times[1:], strict=False))):
        raise ValueError("motion scores require finite canonical17 on a regular 40-Hz GT grid")
    if type(window_offset) is not int or not 0 <= window_offset <= len(times):
        raise ValueError("window_offset must identify an existing grid boundary")
    if len(times) < 40:
        return []
    kernel = np.exp(-.5 * np.arange(-3, 4, dtype=np.float64)**2)
    kernel /= kernel.sum()
    smoothed = np.empty_like(positions)
    padded = np.pad(positions, ((3, 3), (0, 0), (0, 0)), mode="edge")
    for joint in range(17):
        for axis in range(3):
            smoothed[:, joint, axis] = np.convolve(padded[:, joint, axis], kernel, mode="valid")
    velocity = np.gradient(smoothed, .025, axis=0, edge_order=2)
    acceleration = np.linalg.norm(np.gradient(velocity, .025, axis=0, edge_order=2)[:, 1:], axis=2)
    # Smoothing uses +/-3 frames; differentiating twice adds +/-2 support.
    supported = np.convolve(np.pad(validity[:, 1:].all(axis=1).astype(np.int64), (5, 5),
                                   mode="edge"), np.ones(11, dtype=np.int64), mode="valid") == 11
    windows = []
    for start in range(window_offset, len(times) - 39, 40):
        stop = start + 40
        windows.append({
            "start_ns": times[start], "end_ns": times[stop - 1],
            "score_m_s2": float(np.quantile(acceleration[start:stop], .95, method="linear"))
            if supported[start:stop].all() else None,
        })
    return windows


def sequence_scores(sequence, start_ns=0, end_ns=None):
    if sequence.frames[0].source_timestamp_ns != 0:
        raise ValueError("CMU motion-class study requires unchanged zero-origin recordings")
    end_ns = sequence.frames[-1].source_timestamp_ns if end_ns is None else end_ns
    if (type(start_ns) is not int or type(end_ns) is not int or start_ns < 0
            or start_ns > end_ns or end_ns > sequence.frames[-1].source_timestamp_ns):
        raise ValueError("invalid ground-truth class grid bounds")
    # Filter on the WHOLE reference grid before slicing requested metric windows.
    # This is offline labeling of reference motion, not online predictor smoothing.
    times = tuple(range(0, end_ns + 1, 25_000_000))
    lookup = {f.source_timestamp_ns: f for f in sequence.frames}
    if start_ns % 25_000_000 or any(t not in lookup for t in times):
        raise ValueError("reference must cover the exact 40-Hz ground-truth grid")
    frames = [lookup[t] for t in times]
    positions = np.stack([f.joint_positions for f in frames])
    validity = np.stack([f.joint_validity & (f.joint_confidence > 0) for f in frames])
    return window_scores(positions, validity, times, window_offset=start_ns // 25_000_000)


def motion_class(score, smooth_upper, abrupt_lower):
    if (any(type(t) not in (int, float, np.float64) or not np.isfinite(t) or t < 0
            for t in (smooth_upper, abrupt_lower)) or smooth_upper > abrupt_lower):
        raise ValueError("class thresholds must be finite nonnegative and ordered")
    if score is None:
        return "unknown"
    if type(score) not in (int, float, np.float64) or not np.isfinite(score) or score < 0:
        raise ValueError("motion score must be finite nonnegative or unknown")
    if score <= smooth_upper:
        return "smooth"
    return "high-dynamic/abrupt" if score >= abrupt_lower else "medium"


def freeze_classes(index_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("motion-class calibration already exists")
    index_hash = sha256_file(index_path)
    verify_dataset(index_path)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    entries = [e for e in index["recordings"] if e["split"] == "train"]
    training = []
    for entry in entries:
        sequence = load_recording(entry["path"])
        if sequence_split(sequence)[0] != "train":
            raise ValueError("motion-class calibration must use training ground truth only")
        training.append({"sequence_id": sequence.sequence_id, "path": entry["path"],
                         "sha256": entry["sha256"], "windows": sequence_scores(sequence)})
    scores = [w["score_m_s2"] for s in training for w in s["windows"]
              if w["score_m_s2"] is not None]
    if len(scores) < 10:
        raise ValueError("motion-class calibration needs at least ten valid training windows")
    thresholds = np.quantile(scores, [.30, .70], method="linear")
    if sha256_file(index_path) != index_hash:
        raise ValueError("dataset index changed during motion-class calibration")
    report = {
        "format": "poseahead-cmu-motion-classes", "version": 1, "parameters": _PARAMETERS,
        "dataset_index": str(Path(index_path).resolve()), "dataset_index_sha256": index_hash,
        "selection_split": "train", "test_scored": False,
        "smooth_upper_m_s2": float(thresholds[0]), "abrupt_lower_m_s2": float(thresholds[1]),
        "training_valid_windows": len(scores), "training_sequences": training,
        "aggregation": "pooled_nonoverlapping_training_windows_not_independent_people",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def validation_labels(calibration_path):
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if (not isinstance(calibration, dict)
            or calibration.get("format") != "poseahead-cmu-motion-classes"
            or type(calibration.get("version")) is not int or calibration["version"] != 1
            or calibration.get("parameters") != _PARAMETERS
            or calibration.get("selection_split") != "train"):
        raise ValueError("invalid frozen training motion-class calibration")
    index_path = calibration["dataset_index"]
    if sha256_file(index_path) != calibration["dataset_index_sha256"]:
        raise ValueError("calibration dataset index changed")
    verify_dataset(index_path)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    expected_training = []
    for entry in index["recordings"]:
        if entry["split"] == "train":
            expected_training.append({
                "sequence_id": entry["sequence_id"], "path": entry["path"],
                "sha256": entry["sha256"],
                "windows": sequence_scores(load_recording(entry["path"])),
            })
    if calibration["training_sequences"] != expected_training:
        raise ValueError("calibration does not reproduce from training-only references")
    scores = [w["score_m_s2"] for s in calibration["training_sequences"]
              for w in s["windows"] if w["score_m_s2"] is not None]
    thresholds = np.quantile(scores, [.30, .70], method="linear")
    if (calibration["training_valid_windows"] != len(scores)
            or calibration["smooth_upper_m_s2"] != thresholds[0]
            or calibration["abrupt_lower_m_s2"] != thresholds[1]):
        raise ValueError("calibration thresholds do not match saved training scores")
    sequences = []
    for entry in index["recordings"]:
        if entry["split"] != "validation":
            continue
        windows = sequence_scores(load_recording(entry["path"]), start_ns=500_000_000)
        labelled = [{**w, "motion_class": motion_class(w["score_m_s2"], *thresholds)}
                    for w in windows]
        sequences.append({"sequence_id": entry["sequence_id"], "reference_sha256": entry["sha256"],
                          "windows": labelled, "counts": {
                              c: sum(w["motion_class"] == c for w in labelled)
                              for c in ("smooth", "medium", "high-dynamic/abrupt", "unknown")}})
    return {"calibration_sha256": sha256_file(calibration_path), "split": "validation",
            "test_scored": False, "classification_input": "clean_reference_only_not_model_error",
            "sequences": sequences}


def label_validation(calibration_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("motion-class validation labels already exist")
    report = validation_labels(calibration_path)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="fit train-only 30th/70th percentile thresholds")
    freeze.add_argument("index", type=Path)
    freeze.add_argument("output", type=Path)
    label = commands.add_parser("label-validation", help="label GT without choosing models on test")
    label.add_argument("calibration", type=Path)
    label.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        report = freeze_classes(args.index, args.output)
        print(json.dumps({k: report[k] for k in (
            "training_valid_windows", "smooth_upper_m_s2", "abrupt_lower_m_s2", "test_scored",
        )}))
    else:
        report = label_validation(args.calibration, args.output)
        print(json.dumps({s["sequence_id"]: s["counts"] for s in report["sequences"]}))


if __name__ == "__main__":
    main()
