"""Descriptive SPEC E2 stratification using frozen train-only, reference-only motion labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from cmu_motion_classes import validation_labels

from poseahead.artifacts import sha256_file
from poseahead.evaluation import MotionSeries, score_jitter
from poseahead.evaluation.selection import select_from_batches
from poseahead.recording import load_recording


def mean_complete(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def window_metrics(positions, validity, reference, times, clock):
    shared = validity & reference[1]
    error = np.linalg.norm(positions - reference[0], axis=2)[shared]
    jitter = score_jitter(MotionSeries(times, positions, validity, clock),
                         MotionSeries(times, reference[0], reference[1], clock))
    return {"mpjpe_mm": float(np.mean(error) * 1000) if error.size else None,
            "jitter_m_s2": jitter["acceleration_error_m_s2"],
            "pose_coverage": float(shared.mean()), "jitter_coverage": jitter["coverage"]}


def analyze_motion(study, calibration, labels_path, candidate_id="frozen_adaptive"):
    study = Path(study)
    labels = validation_labels(calibration)
    if labels != json.loads(Path(labels_path).read_text(encoding="utf-8")):
        raise ValueError("motion labels differ from frozen training/reference reconstruction")
    result = select_from_batches(study / "selection_protocol.json",
                                 study / "selection_manifest.json")
    if result != json.loads((study / "selection.json").read_text(encoding="utf-8")):
        raise ValueError("selection report does not reproduce")
    if candidate_id not in result["candidate_processors"] or candidate_id == "kalman_cv":
        raise ValueError("motion comparison requires one declared non-baseline candidate")
    labelled = {s["sequence_id"]: s for s in labels["sequences"]}
    manifest = json.loads((study / "selection_manifest.json").read_text(encoding="utf-8"))
    cells = {}
    for batch in manifest["batches"]:
        folder = study / batch["directory"]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        for row in summary["runs"]:
            tag = batch["cells"][row["run_id"]]
            if tag["candidate_id"] not in ("kalman_cv", candidate_id):
                continue
            child = folder / row["directory"]
            config = json.loads((child / "resolved_config.json").read_text(encoding="utf-8"))
            sequence = labelled[row["sequence_id"]]
            if config["reference_sha256"] != sequence["reference_sha256"]:
                raise ValueError("class labels use a different reference")
            truth = load_recording(config["reference"])
            lookup = {f.source_timestamp_ns: f for f in truth.frames}
            with np.load(child / "display_samples.npz", allow_pickle=False) as samples:
                times = tuple(int(t) for t in samples["timestamps_ns"])
                indices = {t: i for i, t in enumerate(times)}
                for window in sequence["windows"]:
                    start = indices.get(window["start_ns"])
                    if (start is None or start + 40 > len(times)
                            or times[start + 39] != window["end_ns"]):
                        raise ValueError("class labels do not align with the fixed metric grid")
                    stops = times[start:start + 40]
                    frames = [lookup[t] for t in stops]
                    reference = (np.stack([f.joint_positions for f in frames]),
                                 np.stack([f.joint_validity & (f.joint_confidence > 0)
                                           for f in frames]))
                    key = (tag["profile_id"], config["seed"], row["sequence_id"],
                           window["motion_class"])
                    cells.setdefault(key, {}).setdefault(tag["candidate_id"], []).append(
                        window_metrics(samples["positions_m"][start:start + 40],
                                       samples["validity"][start:start + 40], reference,
                                       stops, truth.clock_domain))
    metrics = ("mpjpe_mm", "jitter_m_s2", "pose_coverage", "jitter_coverage")
    profiles = {}
    for (profile, seed, sequence, motion), methods in cells.items():
        if set(methods) != {"kalman_cv", candidate_id}:
            raise ValueError("incomplete paired motion-class cell")
        row = {"seed": seed, "sequence_id": sequence,
               "windows": len(methods["kalman_cv"]), **{
                   method: {metric: mean_complete([w[metric] for w in windows])
                            for metric in metrics} for method, windows in methods.items()}}
        profiles.setdefault(profile, {}).setdefault(motion, []).append(row)
    grouped = {}
    for profile, classes in profiles.items():
        grouped[profile] = {}
        for motion, rows in classes.items():
            grouped[profile][motion] = {
                "paired_sequence_seed_cells": len(rows),
                "distinct_sequences": len({r["sequence_id"] for r in rows}),
                "windows_per_seed": sum(r["windows"] for r in rows) // len(
                    {r["seed"] for r in rows}),
                "means": {method: {metric: mean_complete([r[method][metric] for r in rows])
                                   for metric in metrics}
                          for method in ("kalman_cv", candidate_id)},
                "cells": rows,
            }
    return {
        "selection_sha256": sha256_file(study / "selection.json"),
        "calibration_sha256": sha256_file(calibration), "labels_sha256": sha256_file(labels_path),
        "candidate_id": candidate_id, "profiles": grouped, "test_scored": False,
        "advantage_claim": "not_assessed",
        "aggregation": "equal_windows_within_clip_then_equal_clip_and_seed_in_each_present_class",
        "limitation": "exploratory repeated-validation strata; different class clip composition",
        "jitter_scope": "within_each_classified_40-frame_window; not whole-sequence jitter",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", default="frozen_adaptive")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("motion report already exists")
    report = analyze_motion(args.study, args.calibration, args.labels, args.candidate)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"profiles": list(report["profiles"]), "test_scored": False,
                      "advantage_claim": report["advantage_claim"]}))


if __name__ == "__main__":
    main()
