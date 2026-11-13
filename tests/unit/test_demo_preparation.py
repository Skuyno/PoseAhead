"""Offline show preparation verifies evidence before writing any handout."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_test_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_missing_assets_is_read_only(tmp_path, monkeypatch):
    module = script("check_demo", monkeypatch)
    monkeypatch.setattr(module, "load_processor", lambda *args: pytest.fail("Model loading"))
    assert not module.check_demo(tmp_path)["camera_started"]
    report = module.check_demo(tmp_path)
    assert report["status"] == "incomplete"
    assert report["checks"][0]["status"] == "ready"
    assert not report["udp_sent"]
    assert not list(tmp_path.iterdir())
    assert not report["models_loaded_on_cpu"]


def test_preparation_does_not_replace_an_existing_directory(tmp_path, monkeypatch):
    module = script("prepare_demo", monkeypatch)
    with pytest.raises(FileExistsError):
        module.prepare_demo(tmp_path, tmp_path)


def test_failed_preflight_writes_nothing(tmp_path, monkeypatch):
    module = script("prepare_demo", monkeypatch)
    monkeypatch.setattr(module, "check_demo", lambda *args, **kwargs: {"status": "incomplete"})
    with pytest.raises(ValueError, match="preflight"):
        module.prepare_demo(tmp_path, tmp_path / "show")
    assert not list(tmp_path.iterdir())


def test_non_reproducing_selection_is_rejected(tmp_path, monkeypatch):
    module = script("prepare_demo", monkeypatch)
    monkeypatch.setattr(module, "check_demo", lambda *args, **kwargs: {"status": "ready"})
    monkeypatch.setattr(module, "verify_batch", lambda *args: {"status": "verified"})
    monkeypatch.setattr(module, "select_from_batches", lambda *args: {"candidate": "other"})
    first = tmp_path / "outputs" / next(iter(module.PANELS.values()))
    first.mkdir(parents=True)
    (first / "selection.json").write_text(json.dumps({"candidate": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce"):
        module.prepare_demo(tmp_path, tmp_path / "show")
    assert not (tmp_path / "show").exists()


def preflight_files(repository):
    version = repository / "unity/PoseAheadAvatar/ProjectSettings/ProjectVersion.txt"
    version.parent.mkdir(parents=True)
    version.write_text("m_EditorVersion: 6000.3.23f1\n", encoding="utf-8")
    live = repository / "configs/live"
    live.mkdir(parents=True)
    (live / "default.json").write_text("{}", encoding="utf-8")
    (live / "fp16_conv.json").write_text('{"crop_precision":"fp16_conv"}', encoding="utf-8")
    return version, live


def test_preflight_parses_both_camera_profiles_without_opening_camera(tmp_path, monkeypatch):
    module = script("check_demo", monkeypatch)
    preflight_files(tmp_path)
    checks = {item["name"]: item for item in module.check_demo(tmp_path)["checks"]
              if "name" in item}
    assert checks["unity_version"]["version"] == "6000.3.23f1"
    assert checks["live_config"]["crop_precision"] == "autocast"
    assert checks["fast_live_config"]["crop_precision"] == "fp16_conv"


@pytest.mark.parametrize("contents", ("", "m_EditorVersion: 6000.2.0f1\n"))
def test_preflight_rejects_unpinned_or_empty_unity_version(contents, tmp_path, monkeypatch):
    module = script("check_demo", monkeypatch)
    version, _ = preflight_files(tmp_path)
    version.write_text(contents, encoding="utf-8")
    result = next(x for x in module.check_demo(tmp_path)["checks"]
                  if x.get("name") == "unity_version")
    assert result["status"] == "unavailable"
    assert result["reason"]


@pytest.mark.parametrize("filename,contents", (
    ("default.json", "not json"), ("default.json", "[]"),
    ("default.json", '{"crop_precision":"fp16_conv"}'),
    ("fp16_conv.json", "{}"), ("fp16_conv.json", '{"unknown":true}'),
))
def test_preflight_rejects_invalid_or_swapped_profiles(filename, contents, tmp_path, monkeypatch):
    module = script("check_demo", monkeypatch)
    _, live = preflight_files(tmp_path)
    path = live / filename
    path.write_text(contents, encoding="utf-8")
    name = "live_config" if filename == "default.json" else "fast_live_config"
    check = next(x for x in module.check_demo(tmp_path)["checks"] if x.get("name") == name)
    assert check["status"] == "unavailable"
    assert path.read_text(encoding="utf-8") == contents


def mock_evidence(repository, module, monkeypatch):
    monkeypatch.setattr(module, "check_demo", lambda *a, **k: {"status": "ready"})
    monkeypatch.setattr(module, "verify_batch", lambda *a: {"status": "verified"})
    def selected(protocol, manifest):
        return json.loads((protocol.parent / "selection.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(module, "select_from_batches", selected)
    for key, name in module.PANELS.items():
        panel = repository / "outputs" / name
        panel.mkdir(parents=True)
        profiles = (("sigma0", "sigma10", "sigma25", "sigma50") if key == "gaussian"
                    else ("sigma25_dropout002",))
        selected = {"status": "no_eligible_comparator", "candidates": [
            {"candidate_id": candidate, "profiles": {
                profile: {"means": {"display_mpjpe_mm": 1., "jitter_m_s2": 2.}}
                for profile in profiles
            }} for candidate in ("kalman_cv", "frozen_adaptive", "partial_recovery")
        ]}
        for filename, value in (("selection.json", selected), ("selection_protocol.json", {}),
                                ("selection_manifest.json", {})):
            (panel / filename).write_text(json.dumps(value), encoding="utf-8")
    live_path = repository / "outputs/ui_20261005_140958_4ed33ebe/run_report.json"
    live_path.parent.mkdir(parents=True)
    live_path.write_text(json.dumps({
        "producer": {"duration_s": 600., "valid_pose_fps": 18., "sent_frames": 10809,
                     "capture_available_to_send_complete_ms": {"p95": 98.43}},
        "unity": {"received_pose_frames": 10809, "applied_pose_frames": 10809},
    }), encoding="utf-8")


def test_legacy_handout_does_not_require_precision_probes(tmp_path, monkeypatch):
    module = script("prepare_demo", monkeypatch)
    mock_evidence(tmp_path, module, monkeypatch)
    monkeypatch.setattr(module, "verify_probe",
                        lambda *a: pytest.fail("Unexpected precision probe"))
    report = module.prepare_demo(tmp_path, tmp_path / "show")
    assert "recorded_performance" not in report
    assert len(report["verified_batches"]) == 9
    assert report["source_dates"]["live"] == "2026-10-05"


def test_extended_handout_pins_performance_without_live_promotion(tmp_path, monkeypatch):
    module = script("prepare_demo", monkeypatch)
    mock_evidence(tmp_path, module, monkeypatch)
    directories = []
    for name in module.PRECISION_PROBES.values():
        directory = tmp_path / "outputs" / name
        directory.mkdir(parents=True)
        for filename in ("completion.json", "report.json", "poses.npz"):
            (directory / filename).write_bytes(b"test-only source")

    def verified(directory):
        directories.append(directory.name)
        return {"status": "verified", "aggregate_valid_fps": {"autocast": 26., "fp16_conv": 28.},
                "throughput_growth_percent": 7.69, "numeric_identity_all_blocks": True,
                "acceptance_status": "not_assessed"}

    monkeypatch.setattr(module, "verify_probe", verified)
    report = module.prepare_demo(tmp_path, tmp_path / "show", include_performance=True)
    assert directories == list(module.PRECISION_PROBES.values())
    assert len(report["source_sha256"]) == 16
    assert report["recorded_performance"]["recorded_motion"]["acceptance_status"] == "not_assessed"
    assert report["live"]["valid_pose_fps"] == 18.
    assert "not a new measurement" in report["scope"]
    text = (tmp_path / "show/results.md").read_text(encoding="utf-8")
    assert "7.69%" in text and "Это не live-приёмка" in text


def test_failed_precision_verification_leaves_no_handout(tmp_path, monkeypatch):
    module = script("prepare_demo", monkeypatch)
    mock_evidence(tmp_path, module, monkeypatch)

    def corrupted(directory):
        raise ValueError("precision artifact checksum mismatch")

    monkeypatch.setattr(module, "verify_probe", corrupted)
    with pytest.raises(ValueError, match="checksum"):
        module.prepare_demo(tmp_path, tmp_path / "show", include_performance=True)
    assert not (tmp_path / "show").exists()


def test_guide_uses_real_numbering_explicit_styles_and_fixed_tables(tmp_path, monkeypatch):
    module = script("build_demo_guide", monkeypatch)
    source = Path(__file__).resolve().parents[2] / "docs/demo_guide.md"
    output = tmp_path / "guide.docx"
    result = module.build_guide(source, output)
    assert result["status"] == "structurally_verified"
    assert result["visual_qa"].startswith("not_completed")
    with pytest.raises(FileExistsError):
        module.build_guide(source, output)
