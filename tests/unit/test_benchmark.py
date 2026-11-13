import numpy as np
import pytest

from poseahead.core import PoseFrame
from poseahead.runtime.benchmark import BenchmarkStats, latency_summary


def test_nonempty_but_invalid_poses_do_not_inflate_valid_fps() -> None:
    stats = BenchmarkStats()
    invalid = PoseFrame(0, 1, 2, np.zeros((17, 3)), np.zeros(17), np.zeros(17, dtype=bool))
    valid = PoseFrame(1, 3, 4, np.zeros((17, 3)), np.ones(17), np.ones(17, dtype=bool))
    assert not stats.observe(None, 10)
    assert not stats.observe(invalid, 20)
    assert stats.observe(valid, 30)
    report = stats.report(1)
    assert report["processed_fps"] == 3
    assert report["valid_pose_fps"] == 1
    assert report["valid_pose_latency_ms"]["median"] == 30
    assert report["capture_available_to_send_complete_ms"]["p95"] is None
    assert report["acceptance_status"] == "not_assessed"


def test_empty_benchmark_summary_is_json_safe() -> None:
    import json

    report = BenchmarkStats().report(1)
    assert report["all_results_latency_ms"]["median"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_bad_timing_rejected_without_mutating_counters(value):
    stats = BenchmarkStats()
    with pytest.raises(ValueError):
        stats.observe(None, value)
    assert stats.missing_frames == 0 and stats.latency_ms == []
    with pytest.raises(ValueError):
        stats.report(value)
    with pytest.raises(ValueError):
        latency_summary([value])


def test_root_coverage_does_not_confuse_valid_pose_with_valid_translation():
    from dataclasses import replace

    pose = PoseFrame(0, 1, 2, np.zeros((17, 3)), np.ones(17), np.ones(17, dtype=bool))
    stats = BenchmarkStats()
    stats.observe(pose, 10)
    stats.observe(
        replace(
            pose,
            root_position=[0, 0.5, 4],
            metadata={"root_translation_status": "estimated_monocular"},
        ),
        10,
    )
    report = stats.report(1)
    assert report["valid_pose_frames"] == 2
    assert report["valid_pose_with_root_frames"] == report["estimated_root_frames"] == 1
