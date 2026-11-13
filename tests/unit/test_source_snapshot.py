import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def snapshot():
    path = Path(__file__).resolve().parents[2] / "scripts/source_snapshot.py"
    spec = importlib.util.spec_from_file_location("source_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_only_captures_code_and_dependency_declarations(snapshot, tmp_path):
    root = tmp_path / "repo"
    package = root / "src/poseahead"
    package.mkdir(parents=True)
    (package / "module.py").write_bytes(b"VALUE = 1\n")
    (root / ".env").write_text("must not be included")
    output = tmp_path / "snapshot.zip"
    result = snapshot.capture_snapshot(root, output)
    assert result["files"] == 1
    assert result["python_source_sha256"] == snapshot.source_digest({
        "src/poseahead/module.py": b"VALUE = 1\n",
    })
    assert snapshot.verify_snapshot(output) == result
    with ZipFile(output) as archive:
        assert ".env" not in archive.namelist()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        snapshot.capture_snapshot(root, output)
    assert output.read_bytes() == before


def test_snapshot_verifier_rejects_corrupted_payload(snapshot, tmp_path):
    output = tmp_path / "invalid.zip"
    with ZipFile(output, "w") as archive:
        archive.writestr("src/poseahead/a.py", "changed")
        archive.writestr("snapshot_manifest.json", json.dumps({
            "format": "poseahead-research-source-snapshot", "version": 1,
            "sha256": {"src/poseahead/a.py": "0" * 64}, "python_source_sha256": "0" * 64,
        }))
    with pytest.raises(ValueError, match="checksum"):
        snapshot.verify_snapshot(output)
