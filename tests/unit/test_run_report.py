import json
import math

import pytest

from poseahead.runtime.memory import memory_summary, working_set_mib
from poseahead.runtime.run_report import (
    assemble_run_report,
    read_unity_snapshot,
)


def snapshot(tmp_path, **overrides):
    values = dict(
        schema_version=1,
        session_id="42",
        written_unix_ms=10000,
        udp_port=5005,
        received_pose_frames=10,
        applied_pose_frames=7,
        rejected_packets_receiver_total=2,
        rendered_frames=50,
        duration_s=2,
        working_set_mib=100,
        working_set_available=True,
        receiving=True,
        headless=False,
    )
    values.update(overrides)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def read(path, session_id="42"):
    return read_unity_snapshot(path, session_id, not_before_unix_ms=9000, now_unix_ms=11000)


def test_local_process_memory_is_real_or_explicitly_unavailable():
    value = working_set_mib()
    assert value is None or (math.isfinite(value) and value > 0)


@pytest.mark.parametrize(
    "samples,status,growth",
    [
        ([], "unavailable", None),
        ([[0, None], [600, None]], "unavailable", None),
        ([[0, 100], [20, 110]], "insufficient_duration", 10),
        ([[0, 100], [600, 110]], "passed", 10),
        ([[0, 100], [600, 229]], "failed", 129),
    ],
)
def test_memory_acceptance_needs_measurements_and_duration(samples, status, growth):
    result = memory_summary(samples)
    assert result["status"] == status
    assert result["growth_mib"] == growth


def test_exact_session_rendering_and_packet_counts(tmp_path):
    result = read(snapshot(tmp_path))
    assert result["status"] == "observed"
    assert result["render_fps"] == 25
    report = assemble_run_report(
        {"sent_frames": 12}, result, source="camera", requested_duration_s=600
    )
    assert report["transport"]["unconfirmed_pose_frames"] == 2
    assert report["acceptance_status"] == "not_assessed"


def test_headless_cannot_claim_render_fps(tmp_path):
    assert read(snapshot(tmp_path, headless=True))["render_fps"] is None


def test_optional_port_and_sender_count_guards(tmp_path):
    path = snapshot(tmp_path)
    kwargs = dict(not_before_unix_ms=9000, now_unix_ms=11000)
    assert read_unity_snapshot(path, "42", expected_udp_port=5006, **kwargs)["reason"] == (
        "receiver_port_mismatch"
    )
    for count in (9, -1, True, 9.5):
        assert read_unity_snapshot(path, "42", sent_frames=count, **kwargs)["reason"] == (
            "receiver_count_mismatch"
        )
    assert read_unity_snapshot(path, "42", expected_udp_port=5005, sent_frames=10,
                               **kwargs)["status"] == "observed"


def test_configured_render_limit_is_evidence_not_measured_fps(tmp_path):
    result = read(snapshot(
        tmp_path, target_frame_rate=60, v_sync_count=0, demo_frame_limit_active=True,
    ))
    assert result["target_frame_rate"] == 60
    assert result["demo_frame_limit_active"] is True
    assert result["render_fps"] == 25  # Never substitute the configured target.


@pytest.mark.parametrize("values", [
    {"target_frame_rate": True}, {"target_frame_rate": 0}, {"target_frame_rate": -2},
    {"v_sync_count": True}, {"v_sync_count": -1}, {"v_sync_count": 5},
    {"demo_frame_limit_active": "true"}, {"demo_frame_limit_active": True},
    {"target_frame_rate": 120, "v_sync_count": 0, "demo_frame_limit_active": True},
    {"target_frame_rate": 60, "v_sync_count": 1, "demo_frame_limit_active": True},
])
def test_invalid_render_limit_evidence_is_not_accepted(tmp_path, values):
    assert read(snapshot(tmp_path, **values))["status"] == "unavailable"


def test_no_render_callback_means_unmeasured_fps(tmp_path):
    result = read(snapshot(tmp_path, rendered_frames=0))
    assert result["render_fps"] is None
    assert result["render_status"] == "not_assessed"


def test_unavailable_native_memory_is_not_a_measured_zero(tmp_path):
    result = read(snapshot(tmp_path, working_set_available=False, working_set_mib=0))
    assert result["working_set_mib"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"received_pose_frames": True},
        {"received_pose_frames": -1},
        {"applied_pose_frames": 11},
        {"udp_port": 0},
        {"udp_port": 65536},
        {"duration_s": float("nan")},
        {"duration_s": -1},
        {"working_set_mib": float("inf")},
        {"headless": "false"},
        {"written_unix_ms": float("nan")},
        {"session_id": "43"},
        {"written_unix_ms": 8999},
        {"written_unix_ms": 13000},
    ],
)
def test_invalid_stale_or_other_session_never_confirms_receipt(tmp_path, overrides):
    assert read(snapshot(tmp_path, **overrides))["status"] == "unavailable"


def test_missing_partial_or_oversized_snapshot(tmp_path):
    path = tmp_path / "missing.json"
    assert read(path)["status"] == "unavailable"
    path.write_text('{"schema_version":', encoding="utf-8")
    assert read(path)["status"] == "unavailable"
    path.write_text(" " * 65537, encoding="utf-8")
    assert read(path)["status"] == "unavailable"


def test_unity_absence_and_inconsistent_counts_do_not_fabricate_loss():
    producer = {"sent_frames": 3}
    for unity in ({"status": "unavailable"}, {"status": "observed", "received_pose_frames": 4}):
        report = assemble_run_report(producer, unity, source="diagnostic", requested_duration_s=1)
        assert report["transport"]["status"] == "not_assessed"
        assert report["transport"]["unconfirmed_pose_frames"] is None
