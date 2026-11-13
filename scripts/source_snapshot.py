"""Archive research source without recordings, environments, checkpoints or secrets."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

_FILES = ("pyproject.toml", "poetry.lock")
_PATTERNS = {"src/poseahead": "*.py", "scripts": "*.py",
             "configs/processing": "*.json", "requirements": "*.txt"}


def source_digest(files):
    digest = hashlib.sha256()
    for name in sorted(files):
        if name.startswith("src/poseahead/") and name.endswith(".py"):
            digest.update(name.encode() + b"\0")
            digest.update(files[name] + b"\0")
    return digest.hexdigest()


def capture_snapshot(root, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("source snapshot already exists")
    paths = [root / name for name in _FILES if (root / name).is_file()]
    for directory, pattern in _PATTERNS.items():
        paths.extend((root / directory).rglob(pattern))
    files = {}
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("source snapshot rejects symlinks or external targets")
        files[name] = path.read_bytes()
    if not any(name.startswith("src/poseahead/") for name in files):
        raise ValueError("source snapshot needs the poseahead Python package")
    manifest = {
        "format": "poseahead-research-source-snapshot", "version": 1,
        "python_source_sha256": source_digest(files),
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "scope": "Python source/scripts/processing configs/dependency declarations only",
        "excluded": "data/models/recordings/envs/credentials; not a dataset or GPU setup bundle",
    }
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("snapshot_manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
    return verify_snapshot(output)


def verify_snapshot(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate source snapshot entry")
        manifest = json.loads(archive.read("snapshot_manifest.json"))
        if (not isinstance(manifest, dict)
                or manifest.get("format") != "poseahead-research-source-snapshot"
                or type(manifest.get("version")) is not int or manifest["version"] != 1
                or set(names) != {*manifest["sha256"], "snapshot_manifest.json"}):
            raise ValueError("invalid source snapshot manifest")
        files = {}
        for name, expected in manifest["sha256"].items():
            if (not isinstance(name, str) or "\\" in name or ":" in name or name.startswith("/")
                    or any(part in ("", ".", "..") for part in name.split("/"))):
                raise ValueError("invalid source snapshot path")
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("source snapshot checksum mismatch")
            files[name] = data
        if source_digest(files) != manifest["python_source_sha256"]:
            raise ValueError("source snapshot package digest mismatch")
    return {"status": "verified", "files": len(files),
            "python_source_sha256": manifest["python_source_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (verify_snapshot(args.output) if args.verify else
              capture_snapshot(Path(__file__).resolve().parents[1], args.output))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
