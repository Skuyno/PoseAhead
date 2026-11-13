"""Shared byte-level hashing of recordings, checkpoints and experiment artifacts."""

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Hash in bounded chunks, without importing dataset/evaluation/model modules."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
