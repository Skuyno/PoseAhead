"""Presentation presets remain explicit, offline, and independent of live settings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poseahead.artifacts import sha256_file
from poseahead.desktop.demo import DEMO_PRESETS, demo_options


def test_diagnostic_needs_no_recordings_or_models(tmp_path):
    result = demo_options(tmp_path, "diagnostic", udp_port=5100)
    assert result.source == "diagnostic" and result.method == "raw"
    assert result.duration_seconds == 8 and result.udp_port == 5100
    assert not result.record_raw


@pytest.mark.parametrize("preset", DEMO_PRESETS.values())
def test_no_preset_opens_a_camera(preset, tmp_path):
    data = tmp_path / "outputs/cmu_expanded_20260924/13_29.npz"
    data.parent.mkdir(parents=True)
    data.touch()
    raw = tmp_path / "outputs/fixtures/m1_raw_recording.npz"
    raw.parent.mkdir(parents=True)
    raw.touch()
    path = tmp_path / "configs/processing/adaptive_kalman_recovery_pilot.json"
    path.parent.mkdir(parents=True)
    bank = {}
    for horizon in (33, 66, 100, 150):
        checkpoint = tmp_path / "models" / str(horizon)
        checkpoint.mkdir(parents=True)
        for name in ("manifest.json", "model.npz", "normalizer.json"):
            (checkpoint / name).write_text("{}", encoding="utf-8")
        bank[str(horizon)] = {"checkpoint": str(checkpoint),
                              "checkpoint_sha256": sha256_file(checkpoint / "manifest.json")}
    path.write_text(json.dumps({"method": "adaptive", "parameters": {
        "checkpoints": bank, "policy": {},
    }}), encoding="utf-8")
    result = demo_options(tmp_path, preset)
    assert result.source != "camera" and not result.record_raw
    assert result.recording != str(tmp_path / "outputs/cmu_expanded_20260924/14_01.npz")
    if preset == "cmu_adaptive":
        manifest = Path(bank["100"]["checkpoint"]) / "manifest.json"
        manifest.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="Checkpoint"):
            demo_options(tmp_path, preset)


def test_missing_recording_is_not_silently_replaced(tmp_path):
    with pytest.raises(ValueError, match="NPZ"):
        demo_options(tmp_path, "cmu_baseline")


def test_unknown_preset(tmp_path):
    with pytest.raises(ValueError, match="сценарий"):
        demo_options(tmp_path, "camera")


def test_invalid_port(tmp_path):
    with pytest.raises(ValueError, match="UDP"):
        demo_options(tmp_path, "diagnostic", udp_port=0)


def test_apply_demo_does_not_start_or_change_camera_settings(tmp_path):
    from poseahead.desktop.app import METHODS, SOURCES, DesktopApp

    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    app = object.__new__(DesktopApp)
    app.repository = tmp_path
    app.controller = SimpleNamespace(busy=False)
    app.demo_preset = Value(next(iter(DEMO_PRESETS)))
    app.port = Value("5101")
    for key in ("source", "method", "duration", "horizon", "recording",
                "processor_config", "record_raw", "detail", "camera", "live_config"):
        setattr(app, key, Value("unchanged"))
    app.append_log = lambda _: None
    app.apply_demo()
    assert SOURCES[app.source.get()] == "diagnostic"
    assert METHODS[app.method.get()] == "raw"
    assert app.port.get() == "5101"
    assert app.camera.get() == app.live_config.get() == "unchanged"
    app.controller.busy = True
    app.duration.set("123")
    app.apply_demo()
    assert app.duration.get() == "123"
