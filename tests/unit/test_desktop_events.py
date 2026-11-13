"""UI transitions without Tk, a timer, subprocesses or a camera."""

from types import SimpleNamespace

import pytest

from poseahead.desktop.app import DesktopApp


class Value:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


@pytest.fixture
def panel():
    app = object.__new__(DesktopApp)
    app.controller = SimpleNamespace(stop_sent=False)
    app.status, app.detail, app.unity_status = Value(), Value(), Value()
    app.metrics = {key: Value() for key in ("fps", "sent", "latency")}
    app.last_metrics = {}
    app.finished_seen = app.error_seen = False
    app.unity_deadline = None
    app.logs = []
    app.append_log = app.logs.append
    return app


def test_progress_updates_metrics_and_detects_absent_person(panel):
    panel.handle_event(
        {
            "kind": "progress",
            "valid_pose_fps": 0,
            "valid_pose_frames": 0,
            "processed_frames": 10,
            "sent_frames": 0,
        }
    )
    assert panel.metrics["fps"].get() == "0.0"
    assert panel.metrics["sent"].get() == "0"
    assert "Покажите человека" in panel.detail.get()


def test_loading_does_not_overwrite_requested_stop(panel):
    panel.controller.stop_sent = True
    panel.status.set("Остановка…")
    panel.handle_event({"kind": "state", "state": "loading", "message": "loading"})
    assert panel.status.get() == "Остановка…"
    assert panel.logs == ["loading"]


def test_worker_error_is_preserved_after_exit(panel):
    panel.handle_event({"kind": "error", "message": "specific failure"})
    panel.handle_event({"kind": "exit", "code": 1})
    assert panel.error_seen
    assert panel.detail.get() == "specific failure"


def test_exit_without_report_does_not_claim_success(panel):
    panel.handle_event({"kind": "exit", "code": 0})
    assert not panel.finished_seen
    assert panel.status.get() == "Нет итогового отчёта"


@pytest.mark.parametrize(
    "unity,expected",
    [
        ({"status": "unavailable"}, "Нет свежей телеметрии"),
        (
            {
                "status": "observed",
                "received_pose_frames": 10,
                "applied_pose_frames": 7,
                "render_fps": None,
            },
            "FPS не измерен",
        ),
        (
            {
                "status": "observed",
                "received_pose_frames": 10,
                "applied_pose_frames": 7,
                "render_fps": 60,
            },
            "60.0 FPS",
        ),
    ],
)
def test_finished_uses_only_actual_unity_evidence(panel, unity, expected):
    panel.handle_event(
        {
            "kind": "finished",
            "report": {"status": "completed", "sent_frames": 10},
            "run_report": {"unity": unity},
        }
    )
    panel.handle_event({"kind": "exit", "code": 0})
    assert panel.finished_seen
    assert panel.status.get() == "Завершено"
    assert expected in panel.unity_status.get()


def test_unknown_event_is_ignored(panel):
    panel.handle_event({"kind": "future_event"})
    assert panel.logs == []
    assert panel.status.get() == ""


def test_live_receiver_status_expires_using_snapshot_age(panel, monkeypatch):
    clock = SimpleNamespace(monotonic=50.0, unix=100.0)
    monkeypatch.setattr("poseahead.desktop.app.time.monotonic", lambda: clock.monotonic)
    monkeypatch.setattr("poseahead.desktop.app.time.time", lambda: clock.unix)
    panel.handle_event({"kind": "progress", "unity": {
        "status": "observed", "received_pose_frames": 12, "applied_pose_frames": 9,
        "render_fps": 60, "written_unix_ms": 99000, "receiving": True,
    }})
    assert "принято 12" in panel.unity_status.get()
    assert panel.unity_deadline == 51  # already one second old when received
    clock.monotonic = 50.9
    panel.expire_unity_status()
    assert "принято 12" in panel.unity_status.get()
    clock.monotonic = 51
    panel.expire_unity_status()
    assert "перестал подтверждать" in panel.unity_status.get()


def test_unavailable_progress_clears_previous_receipt(panel):
    panel.unity_status.set("Unity: принято 10")
    panel.unity_deadline = 100
    panel.handle_event({"kind": "progress", "unity": {"status": "unavailable"}})
    assert panel.unity_deadline is None
    assert "Нет свежей телеметрии" in panel.unity_status.get()


def test_stage_medians_are_visible_but_do_not_replace_stopping_status(panel):
    event = {"kind": "progress", "pipeline_latency_ms": {
        "frame_age_before_inference_ms": {"median": 12.5},
        "backend_return_ms": {"median": 35.5},
        "recording_and_processing_ms": {"median": None},
    }}
    panel.handle_event(event)
    assert "до 3D: 12.5 мс" in panel.detail.get()
    assert "MeTRAbs: 35.5 мс" in panel.detail.get()
    panel.controller.stop_sent = True
    panel.detail.set("Остановка")
    panel.handle_event(event)
    assert panel.detail.get() == "Остановка"


@pytest.mark.parametrize("extra,expected", [
    ({"receiver_error": "Port busy"}, "Ошибка Unity: Port busy"),
    ({"humanoid_error": "No rig"}, "Ошибка Unity: No rig"),
    ({"receiving": False}, "Receiver не принимает"),
])
def test_receiver_and_rig_failures_are_visible(panel, extra, expected):
    panel.update_unity_status({
        "status": "observed", "received_pose_frames": 12, "applied_pose_frames": 0,
        "render_fps": None, **extra,
    })
    assert expected in panel.unity_status.get()


def test_completed_report_is_historical_evidence_not_expiring_live_status(panel):
    panel.update_unity_status({
        "status": "observed", "received_pose_frames": 12, "applied_pose_frames": 9,
        "render_fps": None,
    })
    assert panel.unity_deadline is None
    panel.expire_unity_status()
    assert "принято 12" in panel.unity_status.get()
