"""A timing report cannot claim exact output equivalence without matching saved arrays."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from poseahead.artifacts import sha256_file


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/benchmark_metrabs_precision.py"
    spec = importlib.util.spec_from_file_location("precision_artifact_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


def fixture(directory, *, altered=False, empty=False):
    report = {"measured_frames_per_block": 1, "runs": [
        {"precision": kind, "metrics": {"valid_pose_frames": 1, "duration_s": duration,
                                         "all_results_latency_ms": {"p95": duration * 1000}}}
        for kind, duration in (("autocast", .04), ("fp16_conv", .03),
                               ("fp16_conv", .03), ("autocast", .04))
    ], "comparisons_to_first_baseline": {
        str(i): {"arrays_exact": True, "validity_and_presence_match": True,
                 "detection_schedule_match": True} for i in range(1, 4)
    }}
    if empty:
        for run in report["runs"]:
            run["metrics"]["valid_pose_frames"] = 0
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    arrays = {}
    for i in range(4):
        arrays.update({
            f"block{i}_present": np.ones(1, dtype=bool),
            f"block{i}_validity": np.ones((1, 17), dtype=bool),
            f"block{i}_detected": np.ones(1, dtype=bool),
            f"block{i}_confidence": np.ones((1, 17), dtype=np.float32),
            f"block{i}_body": np.zeros((1, 17, 3), dtype=np.float32),
            f"block{i}_root": np.zeros((1, 3), dtype=np.float32),
        })
    if altered:
        arrays["block1_body"][0, 1, 0] = .01
    if empty:
        for key, array in arrays.items():
            if key.endswith(("_body", "_root")):
                array[:] = np.nan
            else:
                array[:] = 0
    np.savez_compressed(directory / "poses.npz", **arrays)
    (directory / "completion.json").write_text(json.dumps({
        "format": "poseahead-precision-probe-v1", "sha256": {
            name: sha256_file(directory / name) for name in ("report.json", "poses.npz")
        },
    }), encoding="utf-8")


def test_saved_identity_and_pooled_speed_are_recomputed(tmp_path):
    fixture(tmp_path)
    result = module().verify_probe(tmp_path)
    assert result["numeric_identity_all_blocks"] and result["all_frames_valid"]
    assert result["aggregate_valid_fps"]["autocast"] == 25
    assert result["throughput_growth_percent"] == pytest.approx(100 / 3)
    assert result["acceptance_status"] == "not_assessed"


def test_changed_artifact_rejected(tmp_path):
    fixture(tmp_path)
    with (tmp_path / "report.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        module().verify_probe(tmp_path)


def test_false_identity_claim_rejected_even_with_new_hashes(tmp_path):
    fixture(tmp_path, altered=True)
    with pytest.raises(ValueError, match="identity"):
        module().verify_probe(tmp_path)


def test_no_detected_person_has_no_invented_throughput_gain(tmp_path):
    fixture(tmp_path, empty=True)
    result = module().verify_probe(tmp_path)
    assert not result["all_frames_valid"]
    assert result["throughput_growth_percent"] is None
