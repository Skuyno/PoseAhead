import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.recording import load_recording, save_recording


def test_offline_smoke_preserves_clock_input_and_never_opens_socket(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/smoke_processor_replay.py"
    spec = importlib.util.spec_from_file_location("smoke_processor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*a, **kw):
        raise AssertionError("offline smoke must not open a socket")

    monkeypatch.setattr("socket.socket", forbidden)
    sequence = PoseSequence("detector", "camera-test", tuple(
        PoseFrame(i, i * 50_000_000, i * 50_000_000 + 1, np.zeros((17, 3)),
                  np.ones(17), np.ones(17, bool)) for i in range(5)))
    recording = tmp_path / "recording.npz"
    save_recording(recording, sequence)
    before = recording.read_bytes()
    config = tmp_path / "raw.json"
    config.write_text(json.dumps({"method": "raw", "parameters": {}}))
    output = tmp_path / "smoke"
    report = module.smoke_replay(recording, config, output)
    assert report["input_frames"] == report["output_valid_frames"] == 5
    assert report["udp_algorithms"] == {"RAW": 5}
    assert report["accuracy_assessment"] == "not_scored_no_verified_ground_truth"
    assert recording.read_bytes() == before
    processed = load_recording(output / "processed.npz")
    assert processed.clock_domain == sequence.clock_domain
    assert [f.source_timestamp_ns for f in processed.frames] == [
        f.source_timestamp_ns for f in sequence.frames]
    with pytest.raises(FileExistsError):
        module.smoke_replay(recording, config, output)
