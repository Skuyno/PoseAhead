import hashlib
import subprocess
import sys

import pytest

from poseahead.artifacts import sha256_file


@pytest.mark.parametrize("size", [0, 10, 1024**2 + 7])
def test_file_digest_matches_bytes_and_accepts_strings(tmp_path, size):
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_file(path) == sha256_file(str(path)) == expected


def test_digest_does_not_pull_in_research_or_gpu_layers():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import poseahead.artifacts; "
            "assert not any(name.startswith(('poseahead.datasets', 'poseahead.evaluation', "
            "'poseahead.prediction', 'torch', 'cv2')) for name in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
