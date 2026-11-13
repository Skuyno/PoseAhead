import json

import numpy as np
import pytest

from poseahead.cli import main
from poseahead.datasets.human36m_loader import load_human36m
from poseahead.processing.config import METHODS
from poseahead.recording import load_recording


def fixture_manifest(tmp_path, **overrides):
    points = np.zeros((3, 32, 3))
    points[:] = [1000, 2000, 3000]
    np.save(tmp_path / "points.npy", points)
    manifest = {
        "format": "poseahead-human36m", "version": 1, "subject": "S1",
        "action": "Walking", "take": "1", "camera_id": "cam",
        "positions_file": "points.npy", "coordinate_space": "camera",
        "units": "mm", "fps_numerator": 50, **overrides,
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_cli_import_inspect_and_existing_output_protection(tmp_path, capsys):
    manifest = fixture_manifest(tmp_path)
    output = tmp_path / "recording.npz"
    assert main(["import-h36m", str(manifest), str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 3
    sequence = load_recording(output)
    assert sequence.duration_ns == 40_000_000
    assert sequence.frames[0].metadata["split"] == "train"
    assert main(["inspect-recording", str(output)]) == 0
    capsys.readouterr()
    assert main(["import-h36m", str(manifest), str(output)]) == 2
    assert "Recording error" in capsys.readouterr().err


@pytest.mark.parametrize("method", METHODS)
def test_cli_derived_run_preserves_raw_recording(tmp_path, method):
    manifest = fixture_manifest(tmp_path)
    source = tmp_path / "raw.npz"
    derived = tmp_path / "ema.npz"
    assert main(["import-h36m", str(manifest), str(source)]) == 0
    before = source.read_bytes()
    assert main(["process-recording", str(source), str(derived),
                 "--method", method, "--run-id", "baseline-01"]) == 0
    assert source.read_bytes() == before
    result = load_recording(derived)
    assert result.frames[0].metadata["processor"] == method
    assert result.frames[0].metadata["run_id"] == "baseline-01"
    assert result.frames[1].source_timestamp_ns == 20_000_000


def test_cli_reference_scoring_and_invalid_protocol(tmp_path, capsys):
    manifest = fixture_manifest(tmp_path)
    path = tmp_path / "gt.npz"
    assert main(["import-h36m", str(manifest), str(path)]) == 0
    capsys.readouterr()
    assert main(["score-recording", str(path), str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mpjpe_mm"] == report["root_error_mm"] == 0
    assert report["joint_coverage"] == 1
    assert main(["score-recording", str(path), str(path), "--max-gap-ms", "0"]) == 2
    assert "Scoring error" in capsys.readouterr().err


def test_prepared_npy_rejects_disguised_npz(tmp_path):
    manifest = fixture_manifest(tmp_path)
    with (tmp_path / "points.npy").open("wb") as stream:
        np.savez(stream, positions=np.zeros((3, 32, 3)))
    with pytest.raises(ValueError, match="not an archive"):
        load_human36m(manifest)


def test_world_to_camera_rotation_and_translation(tmp_path):
    manifest = fixture_manifest(
        tmp_path, coordinate_space="world",
        rotation_world_to_camera=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        translation_world_to_camera_mm=[100, 200, 300],
    )
    sequence = load_human36m(manifest)
    np.testing.assert_allclose(sequence.frames[0].root_position, [-1.9, 1.2, 3.3])


@pytest.mark.parametrize("overrides", [
    {"coordinate_space": "world"}, {"units": "m"}, {"subject": "S2"},
    {"coordinate_space": "world", "rotation_world_to_camera": np.eye(3).tolist(),
     "translation_world_to_camera_mm": [0, 0, float("nan")]},
    {"coordinate_space": "world", "rotation_world_to_camera": np.diag([-1, 1, 1]).tolist(),
     "translation_world_to_camera_mm": [0, 0, 0]},
])
def test_import_rejects_ambiguous_geometry_and_invalid_manifest(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_human36m(fixture_manifest(tmp_path, **overrides))
