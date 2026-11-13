"""Run with isolated Python in a fresh base-only wheel installation, outside the checkout."""

import importlib.util
import json
import socket
import sys
from pathlib import Path

import numpy as np

import poseahead
from poseahead.processing.config import create_processor
from poseahead.recording import load_recording, save_recording
from poseahead.runtime.memory import working_set_mib
from poseahead.runtime.unity_replay import diagnostic_sequence
from poseahead.transport import UdpPoseSender, decode_udp_packet


def main():
    root = Path(sys.prefix).resolve()
    package = Path(poseahead.__file__).resolve()
    assert package.is_relative_to(root), "Imported checkout instead of the installed wheel"
    absent = [
        name
        for name in ("torch", "cv2", "pytest", "ruff")
        if importlib.util.find_spec(name) is None
    ]
    assert len(absent) == 4, "Base environment unexpectedly inherited optional/dev dependencies"
    sequence = diagnostic_sequence()
    path = Path.cwd() / "base_probe.npz"
    save_recording(path, sequence)
    restored = load_recording(path)
    assert len(restored.frames) == 160
    methods = ("raw", "ema", "one_euro", "kalman", "kalman_cv")
    for method in methods:
        processor = create_processor(method, {})
        for frame in restored.frames[:20]:
            result = processor.process(frame)
            assert np.all(np.isfinite(result.joint_positions))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2)
        with UdpPoseSender("127.0.0.1", receiver.getsockname()[1]) as sender:
            sender.send(restored.frames[0])
            packet = decode_udp_packet(receiver.recvfrom(2048)[0])
            assert packet.session_id == sender.session_id
    import tkinter

    from poseahead.desktop import app  # lazy UI import: no window or device is opened

    assert app is not None
    print(
        json.dumps(
            {
                "status": "passed",
                "installed_package": str(package),
                "numpy": np.__version__,
                "optional_and_dev_absent": absent,
                "baselines_checked": methods,
                "recording_frames": 160,
                "udp_loopback": "passed",
                "tk_version": tkinter.TkVersion,
                "working_set_mib": working_set_mib(),
                "camera_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
