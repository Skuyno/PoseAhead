"""Worker-to-UI progress uses only matching, fresh local receiver snapshots."""

import json
import threading
import time
from contextlib import contextmanager

import pytest

from poseahead.desktop.session import SessionOptions
from poseahead.desktop.worker import run_session


@pytest.mark.parametrize("source", ["diagnostic", "camera"])
@pytest.mark.parametrize("change,observed", [
    ({}, True), ({"session_id": "other"}, False), ({"udp_port": 5010}, False),
    ({"received_pose_frames": 2}, False), ({"written_unix_ms": 1}, False),
])
def test_live_progress_has_validated_session_telemetry(tmp_path, monkeypatch, source,
                                                    change, observed):
    output = tmp_path / "output"
    output.mkdir()
    live = tmp_path / "live.json"
    live.write_text("{}", encoding="utf-8")
    events = []

    @contextmanager
    def sender(*args):
        class Sender:
            session_id = "42"
        yield Sender()

    def snapshot():
        path = tmp_path / "unity/PoseAheadAvatar/Temp/PoseAheadTelemetry-5005.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            schema_version=1, session_id="42", udp_port=5005,
            written_unix_ms=time.time() * 1000, received_pose_frames=1, applied_pose_frames=1,
            rejected_packets_receiver_total=0, rendered_frames=0, duration_s=.1,
            working_set_mib=100, receiving=True, headless=True, working_set_available=True,
        )
        path.write_text(json.dumps({**data, **change}), encoding="utf-8")

    def replay(*args, on_progress, **kwargs):
        snapshot()
        on_progress({"sent_frames": 1})
        return {"sent_frames": 1, "status": "completed"}

    def webcam(*args, on_progress, **kwargs):
        snapshot()
        on_progress({"kind": "progress", "sent_frames": 1, "sender_session_id": "42"})
        on_progress({"kind": "finished", "report": {
            "sent_frames": 1, "status": "completed", "sender_session_id": "42",
        }})

    monkeypatch.setattr("poseahead.desktop.worker.UdpPoseSender", sender)
    monkeypatch.setattr("poseahead.desktop.worker.stream_sequence", replay)
    monkeypatch.setattr("poseahead.desktop.worker.run_webcam_benchmark", webcam)
    run_session(SessionOptions(source=source, live_config=str(live)),
                tmp_path, output, threading.Event(), events.append)
    progress = next(event for event in events if event["kind"] == "progress")
    assert progress["sender_session_id"] == "42"
    assert (progress["unity"]["status"] == "observed") is observed
    if observed:
        assert progress["unity"]["received_pose_frames"] == 1
        assert progress["unity"]["render_fps"] is None  # headless never claims a render FPS
    finished = next(event for event in events if event["kind"] == "finished")
    assert (finished["run_report"]["unity"]["status"] == "observed") is observed
