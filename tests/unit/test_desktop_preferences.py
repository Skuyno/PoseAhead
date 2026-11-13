"""Preferences preserve configuration, never consent or corrupted user files."""

import json
from types import SimpleNamespace

import pytest

from poseahead.desktop.app import METHODS, SOURCES, DesktopApp
from poseahead.desktop.preferences import PreferenceStore, validate_preferences
from poseahead.desktop.session import SessionOptions


def test_roundtrip_does_not_require_camera_configs_or_models(tmp_path):
    store = PreferenceStore(tmp_path / "outputs/desktop_settings.json")
    assert store.load() == ({}, None)
    values = {
        "source": "camera", "method": "config", "duration_seconds": 600.0,
        "camera_index": 2, "udp_port": 5010, "horizon_ms": 66,
        "recording": "несуществующий.npz", "live_config": "missing.json",
        "processor_config": "missing-models.json", "unity_editor": "F:/Unity/Unity.exe",
    }
    store.save(values)
    assert store.load() == (values, None)
    assert not list(store.path.parent.glob("*.tmp"))


@pytest.mark.parametrize("values", [
    [], {"record_raw": True}, {"auto_start": True}, {"source": "video"},
    {"source": []}, {"method": "adaptive"}, {"method": None},
    {"duration_seconds": 0}, {"duration_seconds": True}, {"duration_seconds": float("nan")},
    {"duration_seconds": float("inf")}, {"duration_seconds": 3601},
    {"camera_index": -1}, {"camera_index": True}, {"udp_port": 0},
    {"udp_port": 65536}, {"udp_port": 5.5}, {"horizon_ms": True}, {"horizon_ms": 99},
    {"recording": 2}, {"live_config": "\0"}, {"unity_editor": "x" * 8193},
], ids=lambda _: "invalid")
def test_invalid_preferences_are_rejected(values):
    with pytest.raises(ValueError):
        validate_preferences(values)


@pytest.mark.parametrize("data", [
    b"{unfinished", b"x" * 65537, b"[]", b'{"version":true,"settings":{}}',
    b'{"version":2,"settings":{}}', b'{"version":1,"settings":{"record_raw":true}}',
], ids=lambda data: "bytes-" + str(len(data)))
def test_bad_file_is_preserved_and_does_not_prevent_default_startup(tmp_path, data):
    path = tmp_path / "settings.json"
    path.write_bytes(data)
    store = PreferenceStore(path)
    values, warning = store.load()
    assert values == {} and warning
    with pytest.raises(ValueError, match="без изменений"):
        store.save({"source": "diagnostic"})
    assert path.read_bytes() == data
    path.unlink()
    assert store.load() == ({}, None)
    store.save({"source": "diagnostic"})


def test_atomic_replace_failure_preserves_last_good_file(tmp_path, monkeypatch):
    store = PreferenceStore(tmp_path / "settings.json")
    store.save({"udp_port": 5005})
    original = store.path.read_bytes()

    def failed_replace(*args):
        raise OSError("simulated sharing violation")

    monkeypatch.setattr("poseahead.desktop.preferences.os.replace", failed_replace)
    with pytest.raises(OSError, match="sharing"):
        store.save({"udp_port": 5100})
    assert store.path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [store.path]


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_app_restores_camera_choices_without_starting_or_enabling_recording(tmp_path):
    app = object.__new__(DesktopApp)
    app.preferences = PreferenceStore(tmp_path / "settings.json")
    app.preferences.save({"source": "camera", "method": "kalman_cv", "camera_index": 2,
                          "duration_seconds": 60, "udp_port": 5100})
    for key in ("source", "method", "duration", "camera", "port", "horizon", "recording",
                "live_config", "processor_config", "unity_editor", "record_raw"):
        setattr(app, key, Value())
    app.controller = SimpleNamespace(busy=False)
    app.append_log = lambda _: None
    app.restore_settings()  # controller deliberately has no start method
    assert SOURCES[app.source.get()] == "camera"
    assert METHODS[app.method.get()] == "kalman_cv"
    assert app.camera.get() == "2" and app.port.get() == "5100"
    assert not app.record_raw.get()


def test_app_save_does_not_persist_recording_consent(tmp_path):
    app = object.__new__(DesktopApp)
    app.preferences = PreferenceStore(tmp_path / "settings.json")
    app.options = lambda: SessionOptions(source="camera", record_raw=True)
    app.unity_editor = Value("F:/Unity/Unity.exe")
    app.save_settings()
    settings = json.loads(app.preferences.path.read_text(encoding="utf-8"))["settings"]
    assert settings["source"] == "camera"
    assert "record_raw" not in settings and "auto_start" not in settings
