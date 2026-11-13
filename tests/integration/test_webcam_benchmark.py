"""Exercise live orchestration without camera, GPU, downloads, or external traffic."""

from __future__ import annotations

import json
import runpy
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from poseahead.core import PoseFrame
from poseahead.processing.learned import LearnedProcessor
from poseahead.recording import load_recording
from poseahead.runtime.camera import CameraFrame
from poseahead.transport import UdpPoseSender, decode_udp_packet
from poseahead.transport.udp_protocol import Algorithm

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_webcam.py"


def test_compatibility_launcher_reuses_packaged_camera_runtime():
    from poseahead.runtime.webcam import main

    assert runpy.run_path(str(SCRIPT))["main"] is main


@pytest.mark.parametrize("method, algorithm, horizon", [
    ("raw", Algorithm.RAW, 0), ("ema", Algorithm.EMA, 0),
    ("one_euro", Algorithm.ONE_EURO, 0), ("kalman", Algorithm.KALMAN, 0),
    ("kalman_cv", Algorithm.KALMAN_EXTRAPOLATION, 100),
])
def test_live_udp_identifies_method_and_prediction_target(live_probe, method, algorithm, horizon):
    run, _ = live_probe
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.1)
        run(receiver.getsockname()[1], "--send-udp", "--processor", method)
        packet = decode_udp_packet(receiver.recvfrom(1200)[0])
        assert packet.algorithm == algorithm
        assert packet.target_offset_ms == horizon
        assert packet.frame.source_timestamp_ns == 1_021_000_000


def test_fractional_udp_horizon_rejected_before_camera(live_probe, tmp_path):
    run, state = live_probe
    config = tmp_path / "fractional.json"
    config.write_text(json.dumps({
        "method": "kalman_cv", "parameters": {"target_offset_ms": 33.5},
    }), encoding="utf-8")
    with pytest.raises(SystemExit):
        run(5005, "--send-udp", "--processor-config", str(config))
    assert not hasattr(state, "backend_config")


@pytest.mark.parametrize("method", ["learned", "adaptive"])
def test_learned_method_is_not_silently_sent_as_a_raw_live_packet(live_probe, monkeypatch, method):
    run, state = live_probe
    monkeypatch.setitem(state.main.__globals__, "create_processor",
                        lambda *args: SimpleNamespace(method=method, history_sampling="exact"))
    with pytest.raises(SystemExit):
        run(5005, "--send-udp")
    assert not hasattr(state, "backend_config")
    assert state.sender is None


def test_live_learned_udp_warmup_then_prediction(live_probe, monkeypatch, tmp_path):
    run, state = live_probe
    state.mode = "valid"

    class Predictor:
        history_frames, step_ns, horizon_ms = 2, 5_000_000, 33

        def predict(self, positions, times):
            assert times[1] - times[0] == self.step_ns
            return positions[-1]

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("poseahead.processing.learned.FixedHorizonPredictor", lambda _: Predictor())
    processor = LearnedProcessor(checkpoint=tmp_path, history_sampling="linear_past")
    monkeypatch.setitem(state.main.__globals__, "create_processor", lambda *args: processor)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(.1)
        report = run(receiver.getsockname()[1], "--send-udp")
        packets = [decode_udp_packet(receiver.recvfrom(1200)[0]) for _ in range(3)]
    assert [p.algorithm for p in packets] == [Algorithm.RAW, Algorithm.PREDICTOR_33_MS,
                                            Algorithm.PREDICTOR_33_MS]
    assert [p.target_offset_ms for p in packets] == [0, 33, 33]
    assert report["prediction_status_counts"] == {"warmup": 1, "predicted": 2}
    assert report["processor_parameters"]["history_sampling"] == "linear_past"
    assert state.camera_closed and state.sender_closed


def test_live_records_raw_partial_and_missing_observations(live_probe, tmp_path):
    run, state = live_probe
    path = tmp_path / "raw.npz"
    report = run(5005, "--record-output", str(path), "--processor", "ema")
    sequence = load_recording(path)
    assert [frame.frame_id for frame in sequence.frames] == [1, 2, 3]
    assert sequence.frames[0].is_valid
    assert not sequence.frames[1].is_valid
    assert not sequence.frames[2].joint_validity.any()
    assert sequence.frames[2].metadata["observation_status"] == "missing"
    assert all("processor" not in frame.metadata for frame in sequence.frames)
    assert report["processor"] == "ema"
    assert state.camera_closed


@pytest.fixture
def live_probe(monkeypatch, tmp_path):
    # Patch the packaged runtime, restoring globals between tests. Transport stays real.
    clock = SimpleNamespace(ns=1_000_000_000)
    state = SimpleNamespace(captured=0, camera_closed=False, sender=None, mode="mixed")

    class Camera:
        def __init__(self, index):
            state.camera_index = index

        def __enter__(self):
            return self

        def __exit__(self, *_):
            state.camera_closed = True

        def get(self, timeout):
            clock.ns += 10_000_000
            if state.mode == "empty":
                return None
            frame = CameraFrame(state.captured, clock.ns, np.zeros((2, 2, 3), dtype=np.uint8))
            state.captured += 1
            return frame

        def snapshot(self):
            return state.captured, 0, 0

        def discard_pending_and_snapshot(self):
            return self.snapshot()

    class Backend:
        def __init__(self, **kwargs):
            state.startup_callback = kwargs.pop("on_startup")
            state.stop_callback = kwargs.pop("stop_requested")
            state.backend_config = kwargs
            state.startup_callback("Проверка исходников MeTRAbs")
            if state.stop_callback():
                from poseahead.perception.metrabs import MetrabsStartupCancelled

                raise MetrabsStartupCancelled()

        def infer(self, image, *, frame_id, source_timestamp_ns):
            clock.ns += 1_000_000
            if state.mode == "failure":
                raise RuntimeError("synthetic inference failure")
            if frame_id == 3 and state.mode == "mixed":
                return None
            validity = np.ones(17, dtype=bool)
            if frame_id == 2 and state.mode == "mixed":
                validity[-1] = False
            return PoseFrame(
                frame_id, source_timestamp_ns, clock.ns,
                np.zeros((17, 3)), np.ones(17), validity,
            )

    class Sender(UdpPoseSender):
        def __init__(self, *args):
            super().__init__(*args)
            state.sender = self
            state.sender_closed = False

        def close(self):
            super().close()
            state.sender_closed = True

    cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        memory_allocated=lambda: 1024**2,
        memory_reserved=lambda: 2 * 1024**2,
        max_memory_allocated=lambda: 1024**2,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    namespace = runpy.run_path(str(SCRIPT))
    main = namespace["main"]
    state.main = main
    replacements = dict(
        time=SimpleNamespace(
            perf_counter=lambda: clock.ns / 1e9, perf_counter_ns=lambda: clock.ns,
        ),
        LatestFrameCamera=Camera,
        MetrabsBackend=Backend,
        UdpPoseSender=Sender,
    )
    for key, value in replacements.items():
        monkeypatch.setitem(main.__globals__, key, value)
    config = tmp_path / "live.json"
    output = tmp_path / "report.json"

    def run(port, *extra, **hooks):
        config.write_text(json.dumps({"udp_port": port, "camera_index": 7}), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            str(SCRIPT), "--config", str(config), "--duration", "0.03",
            "--warmup", "1", "--output", str(output), *extra,
        ])
        assert main(**hooks) == 0
        return json.loads(output.read_text(encoding="utf-8"))

    return run, state


@pytest.mark.parametrize("send", [False, True])
def test_live_report_and_optional_real_loopback(live_probe, send):
    run, state = live_probe
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.05)
        report = run(receiver.getsockname()[1], *(["--send-udp"] if send else []))
        if send:
            packet = decode_udp_packet(receiver.recvfrom(1200)[0])
            assert packet.frame.frame_id == 1  # warm-up frame 0 was never sent
            assert state.sender_closed
        else:
            assert state.sender is None
        with pytest.raises(socket.timeout):
            receiver.recvfrom(1200)  # neither partial frame 2 nor missing frame 3 was sent

    assert state.camera_closed
    assert state.camera_index == 7
    assert state.backend_config == {
        "confidence_threshold": 0.3,
        "num_aug": 1,
        "internal_batch_size": 8,
        "detector_interval": 10,
        "max_input_side": 640,
        "default_fov_degrees": 55.0,
        "camera_intrinsics": None,
        "crop_precision": "autocast",
    }
    assert report["captured_frames"] == report["processed_frames"] == 3
    assert report["valid_pose_frames"] == 1
    assert report["partial_or_invalid_pose_frames"] == report["missing_pose_frames"] == 1
    assert report["valid_pose_fps"] == pytest.approx(1 / report["duration_s"])
    assert report["valid_pose_latency_ms"]["p95"] == 1
    assert report["capture_available_to_send_complete_ms"]["samples"] == int(send)
    assert report["udp_enabled"] == send
    assert report["acceptance_status"] == "not_assessed"
    assert report["camera_images_stored"] is False
    assert report["sent_frames"] == int(send)
    stages = report["pipeline_latency_ms"]
    assert stages["backend_return_ms"]["samples"] == 3
    assert stages["backend_return_ms"]["median"] == 1
    assert stages["capture_available_to_send_complete_ms"]["samples"] == int(send)
    assert (report["sender_session_id"] is not None) == send
    assert report["host_memory"]["status"] in ("insufficient_duration", "unavailable")
    assert len(report["host_memory"]["samples_seconds_working_set_mib"]) >= 2
    assert report["gpu_reserved_growth_mib"] == 0
    assert report["gpu_reserved_stability_status"] == "insufficient_duration"


def test_no_camera_frames_produces_finite_empty_report(live_probe):
    run, state = live_probe
    state.mode = "empty"
    report = run(5005, "--warmup", "0")
    assert state.camera_closed
    assert report["processed_frames"] == report["captured_frames"] == 0
    assert report["valid_pose_fps"] == 0
    assert report["valid_pose_latency_ms"]["p95"] is None
    assert len(report["host_memory"]["samples_seconds_working_set_mib"]) >= 2
    assert report["pipeline_latency_ms"]["backend_return_ms"]["samples"] == 0


def test_fast_precision_config_reaches_backend(live_probe, tmp_path):
    run, state = live_probe
    path = tmp_path / "fast.json"
    path.write_text(json.dumps({"crop_precision": "fp16_conv"}), encoding="utf-8")
    report = run(5005, "--config", str(path))
    assert state.backend_config["crop_precision"] == "fp16_conv"
    assert report["resolved_config"]["crop_precision"] == "fp16_conv"


@pytest.mark.parametrize("mode, message", [
    ("empty", "camera delivered no frame during warmup"),
    ("failure", "synthetic inference failure"),
])
def test_failure_closes_camera_and_udp_sender(live_probe, mode, message):
    run, state = live_probe
    state.mode = mode
    with pytest.raises(RuntimeError, match=message):
        run(5005, "--send-udp")
    assert state.camera_closed
    assert state.sender_closed


@pytest.mark.parametrize("duration", ["nan", "inf", "-inf", "0", "-1"])
def test_invalid_duration_rejected_before_backend_load(live_probe, duration):
    run, state = live_probe
    with pytest.raises(SystemExit) as error:
        run(5005, f"--duration={duration}")
    assert error.value.code == 2
    assert not hasattr(state, "backend_config")
    assert state.sender is None


def test_ui_graceful_stop_saves_raw_prefix_and_closes_resources(live_probe, tmp_path):
    run, state = live_probe
    events = []
    stopped = False

    def progress(event):
        nonlocal stopped
        events.append(event)
        if event["kind"] == "progress":
            stopped = True

    path = tmp_path / "stopped.npz"
    report = run(5005, "--send-udp", "--record-output", str(path),
                 stop_requested=lambda: stopped, on_progress=progress)
    assert report["status"] == "stopped"
    assert report["processed_frames"] == 1
    assert len(load_recording(path).frames) == 1
    assert state.camera_closed and state.sender_closed
    assert [event["kind"] for event in events][-2:] == ["progress", "finished"]
    assert events[-2]["sender_session_id"] == str(state.sender.session_id)
    assert events[-2]["sent_frames"] == 1
    assert all(event["kind"] == "state" for event in events[:-2])
    messages = [event["message"] for event in events[:-2]]
    assert "Открытие камеры 7" in messages
    assert "Прогрев: кадр 1/1 — ожидание кадра камеры" in messages
    assert "Прогрев: кадр 1/1 — оценка позы" in messages


def test_ui_cancel_during_warmup_does_not_fabricate_a_recording(live_probe, tmp_path, monkeypatch):
    _, state = live_probe
    path = tmp_path / "not-recorded.npz"
    events = []
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--send-udp", "--record-output", str(path)])
    assert state.main(stop_requested=lambda: state.captured > 0, on_progress=events.append) == 0
    assert events[-1]["report"]["status"] == "stopped_before_measurement"
    assert not path.exists()
    assert state.camera_closed and state.sender_closed


@pytest.mark.parametrize("stage", [
    "Загрузка библиотеки PyTorch", "Проверка исходников MeTRAbs", "Открытие камеры 7",
])
def test_ui_cancel_at_startup_boundary_never_opens_camera(live_probe, stage):
    _, state = live_probe
    events = []
    stopped = False

    def progress(event):
        nonlocal stopped
        events.append(event)
        if event.get("message") == stage:
            stopped = True

    assert state.main(
        ["--camera", "7", "--send-udp"],
        stop_requested=lambda: stopped, on_progress=progress,
    ) == 0
    assert events[-1]["report"]["status"] == "stopped_before_measurement"
    assert not hasattr(state, "camera_index")
    if state.sender is not None:
        assert state.sender_closed


def test_camera_open_failure_is_labeled_as_camera_not_model_loading(live_probe, monkeypatch):
    _, state = live_probe
    events = []

    def fail_camera(index):
        raise RuntimeError("synthetic camera open failure")

    monkeypatch.setitem(state.main.__globals__, "LatestFrameCamera", fail_camera)
    with pytest.raises(RuntimeError, match="camera open failure"):
        state.main(["--camera", "7", "--send-udp"], on_progress=events.append)
    assert events[-1]["message"] == "Открытие камеры 7"
    assert state.sender_closed


def test_empty_camera_record_request_does_not_write_invalid_empty_npz(live_probe, tmp_path):
    run, state = live_probe
    state.mode = "empty"
    path = tmp_path / "empty.npz"
    report = run(5005, "--warmup", "0", "--record-output", str(path))
    assert report["raw_pose_recording"] is None
    assert not path.exists()
