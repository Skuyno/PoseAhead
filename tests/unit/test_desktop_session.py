import json
import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from poseahead.desktop.session import SessionController, SessionOptions, unity_launch_command
from poseahead.desktop.worker import run_session
from poseahead.recording import save_recording
from poseahead.runtime.unity_replay import diagnostic_sequence
from poseahead.transport import decode_udp_packet


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "video"),
        ("method", "adaptive"),
        ("duration_seconds", 0),
        ("duration_seconds", True),
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("duration_seconds", 3601),
        ("camera_index", -1),
        ("camera_index", True),
        ("udp_port", 0),
        ("udp_port", 65536),
        ("udp_port", 1.5),
        ("horizon_ms", 101),
        ("horizon_ms", True),
        ("record_raw", True),
        ("record_raw", "yes"),
    ],
)
def test_invalid_ui_request_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(SessionOptions(), **{field: value}).validate()


def test_camera_config_cannot_send_to_an_external_address(tmp_path):
    config = tmp_path / "live.json"
    config.write_text(json.dumps({"udp_host": "192.0.2.1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="127.0.0.1"):
        SessionOptions(source="camera", live_config=str(config)).validate()


def test_missing_recording_is_rejected():
    with pytest.raises(ValueError, match="NPZ"):
        SessionOptions(source="recording").validate()


@pytest.mark.parametrize(
    "values",
    [
        [],
        {},
        {"method": "adaptive", "parameters": {}},
        {
            "method": "raw",
            "parameters": [],
        },
        {"method": "raw", "parameters": {}, "extra": 1},
    ],
)
def test_invalid_custom_processor_config(tmp_path, values):
    config = tmp_path / "processor.json"
    config.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError):
        SessionOptions(method="config", processor_config=str(config)).validate()


@pytest.mark.parametrize("method", ["learned", "adaptive"])
def test_live_mlp_requires_explicit_causal_history(tmp_path, method):
    live = tmp_path / "live.json"
    live.write_text("{}", encoding="utf-8")
    processor = tmp_path / "processor.json"
    parameters = {"checkpoints": {}, "policy": {}} if method == "adaptive" else {}
    processor.write_text(json.dumps({"method": method, "parameters": parameters}), encoding="utf-8")
    with pytest.raises(ValueError, match="linear_past"):
        SessionOptions(
            source="camera", method="config", live_config=str(live), processor_config=str(processor)
        ).validate()


def finish(controller):
    deadline = time.monotonic() + 10
    while controller.busy:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    events = []
    while not controller.events.empty():
        events.append(controller.events.get_nowait())
    return events


def repository(tmp_path):
    # A project marker suffices; a producer must not depend on loose checkout scripts.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "poseahead"\n', encoding="utf-8")
    return tmp_path


def test_actual_ui_worker_uses_udp_and_new_artifact_directory(tmp_path):
    controller = SessionController(repository(tmp_path))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.05)
        controller.start(SessionOptions(udp_port=receiver.getsockname()[1], duration_seconds=0.5))
        with pytest.raises(RuntimeError, match="уже"):
            controller.start(SessionOptions())
        packets = []
        deadline = time.monotonic() + 10
        while controller.busy:
            assert time.monotonic() < deadline
            try:
                packets.append(decode_udp_packet(receiver.recvfrom(1200)[0]))
            except TimeoutError:
                pass
        while True:
            try:
                packets.append(decode_udp_packet(receiver.recvfrom(1200)[0]))
            except TimeoutError:
                break
    events = finish(controller)
    assert len(packets) >= 2
    assert [packet.frame.frame_id for packet in packets] == list(range(len(packets)))
    assert events[-1] == {"kind": "exit", "code": 0}
    report = json.loads((controller.output_directory / "report.json").read_text(encoding="utf-8"))
    assert report["sent_frames"] == len(packets)
    assert report["status"] == "completed"
    assert report["scope"] == "display replay, not measured camera latency"
    combined = json.loads((controller.output_directory / "run_report.json").read_text("utf-8"))
    assert combined["producer"] == report
    assert combined["unity"]["status"] == "unavailable"
    assert combined["transport"]["received_pose_frames"] is None
    assert combined["acceptance_status"] == "not_assessed"
    assert (controller.output_directory / "request.json").is_file()
    assert (controller.output_directory / "worker.log").is_file()
    assert not (tmp_path / "scripts").exists()
    first = controller.output_directory
    original = (first / "report.json").read_bytes()
    controller.start(SessionOptions(duration_seconds=0.01))
    finish(controller)
    assert first != controller.output_directory
    assert (first / "report.json").read_bytes() == original


def test_stop_is_graceful_and_stops_only_owned_producer(tmp_path):
    controller = SessionController(repository(tmp_path))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(5)
            controller.start(SessionOptions(udp_port=receiver.getsockname()[1]))
            receiver.recvfrom(1200)
            controller.request_stop()
            controller.request_stop()  # Idempotent.
        events = finish(controller)
        assert controller.stop_sent
        assert events[-1] == {"kind": "exit", "code": 0}
        report = json.loads(
            (controller.output_directory / "report.json").read_text(encoding="utf-8")
        )
        assert report["status"] == "stopped"
        assert 1 <= report["sent_frames"] < 160
    finally:
        controller.force_stop()
        finish(controller)


def test_bounded_ui_event_queue_keeps_recent_events(tmp_path):
    controller = SessionController(tmp_path)
    for index in range(1000):
        controller._publish({"kind": "log", "message": str(index)})
    controller._publish({"kind": "exit", "code": 0})
    assert controller.events.qsize() == 200
    events = [controller.events.get_nowait() for _ in range(200)]
    assert events[-1] == {"kind": "exit", "code": 0}


def test_validation_does_not_launch_a_process_or_make_artifacts(tmp_path):
    controller = SessionController(tmp_path)
    with pytest.raises(ValueError):
        controller.start(SessionOptions(udp_port=0))
    assert controller.process is None
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "version,valid",
    [("6000.3.23f1_09d2ecc7fb28", True), ("6000.3.23f1", True), ("6000.2.0f1", False)],
)
def test_unity_launch_preflight_rejects_other_versions(tmp_path, monkeypatch, version, valid):
    editor = tmp_path / "Unity.exe"
    editor.write_text("fake metadata-only fixture", encoding="utf-8")
    settings = tmp_path / "unity/PoseAheadAvatar/ProjectSettings"
    settings.mkdir(parents=True)
    (settings / "ProjectVersion.txt").write_text("m_EditorVersion: 6000.3.23f1\n", encoding="utf-8")
    monkeypatch.setattr(
        "poseahead.desktop.session.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=version),
    )
    if valid:
        assert unity_launch_command(tmp_path, editor) == [
            str(editor),
            "-projectPath",
            str(settings.parent),
        ]
    else:
        with pytest.raises(ValueError, match="6000.3.23f1"):
            unity_launch_command(tmp_path, editor)


def test_npz_worker_preserves_the_original_recording(tmp_path):
    checkout = repository(tmp_path)
    recording = checkout / "source.npz"
    save_recording(recording, diagnostic_sequence())
    original = recording.read_bytes()
    controller = SessionController(checkout)
    controller.start(
        SessionOptions(source="recording", recording=str(recording), duration_seconds=0.1)
    )
    events = finish(controller)
    assert events[-1]["code"] == 0
    report = json.loads((controller.output_directory / "report.json").read_text(encoding="utf-8"))
    assert report["sequence_id"] == "unity-diagnostic"
    assert report["sent_frames"] > 0
    assert recording.read_bytes() == original


def test_worker_errors_are_visible_and_not_reported_as_success(tmp_path):
    checkout = repository(tmp_path)
    config = checkout / "bad.json"
    config.write_text(json.dumps({"method": "raw", "parameters": {"unknown": 1}}), encoding="utf-8")
    controller = SessionController(checkout)
    controller.start(SessionOptions(method="config", processor_config=str(config)))
    events = finish(controller)
    assert events[-1]["code"] == 1
    assert any(event["kind"] == "error" for event in events)
    assert not (controller.output_directory / "report.json").exists()
    assert "Traceback" in (controller.output_directory / "worker.log").read_text(encoding="utf-8")


def test_custom_processor_snapshot_resolves_paths_without_changing_original(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    config = configs / "learned.json"
    original = json.dumps({"method": "learned", "parameters": {"checkpoint": "../model"}})
    config.write_text(original, encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        "poseahead.desktop.worker.load_processor",
        lambda _: SimpleNamespace(method="learned", parameters={}),
    )
    monkeypatch.setattr(
        "poseahead.desktop.worker.stream_sequence",
        lambda *args, **kwargs: {"sent_frames": 0, "status": "completed"},
    )
    run_session(
        SessionOptions(method="config", processor_config=str(config)),
        tmp_path,
        output,
        threading.Event(),
        lambda _: None,
    )
    snapshot = json.loads((output / "processor.json").read_text(encoding="utf-8"))
    assert snapshot["parameters"]["checkpoint"] == str((tmp_path / "model").resolve())
    assert config.read_text(encoding="utf-8") == original
