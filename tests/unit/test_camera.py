"""Test the real capture thread with a controlled device, never a physical camera."""

import sys
import threading
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from poseahead.runtime.camera import LatestFrameCamera


@pytest.fixture
def controlled_camera(monkeypatch):
    class Capture:
        def __init__(self):
            self.entered = threading.Event()
            self.unblock = threading.Event()
            self.active_read = False
            self.release_count = 0
            self.success = True

        def isOpened(self):
            return True

        def set(self, *_):
            return True

        def read(self):
            self.active_read = True
            self.entered.set()
            self.unblock.wait(5)
            self.active_read = False
            return self.success, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            assert not self.active_read, "release raced an active read"
            self.release_count += 1

    capture = Capture()
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=lambda _: capture, CAP_PROP_BUFFERSIZE=1,
    ))
    camera = LatestFrameCamera()
    yield camera, capture
    capture.unblock.set()
    camera.close(timeout=1)


def test_blocked_read_reports_shutdown_failure_without_racing_release(controlled_camera):
    camera, capture = controlled_camera
    camera.start()
    assert capture.entered.wait(1)
    with pytest.raises(RuntimeError, match="shutdown timed out"):
        camera.close(timeout=0.01)
    assert camera._thread.is_alive()
    assert capture.release_count == 0
    assert camera.get(timeout=0) is None
    capture.unblock.set()
    camera.close(timeout=1)
    assert not camera._thread.is_alive()
    assert capture.release_count == 1
    assert camera.snapshot() == (0, 0, 0)  # in-flight read was never published after close


def test_close_before_start_and_repeated_close_release_exactly_once(controlled_camera):
    camera, capture = controlled_camera
    camera.close()
    camera.close()
    assert capture.release_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        camera.start()


def test_close_wakes_waiting_consumer_before_shutdown_timeout(controlled_camera):
    camera, capture = controlled_camera
    camera.start()
    assert capture.entered.wait(1)
    received = []
    consumer = threading.Thread(target=lambda: received.append(camera.get(timeout=2)))
    consumer.start()
    try:
        with pytest.raises(RuntimeError, match="shutdown timed out"):
            camera.close(timeout=0.01)
        consumer.join(timeout=1)
        assert not consumer.is_alive()
        assert received == [None]
    finally:
        capture.unblock.set()
        consumer.join(timeout=2)


def test_capture_failure_reaches_consumer_and_releases_device(controlled_camera):
    camera, capture = controlled_camera
    capture.success = False
    capture.unblock.set()
    camera.start()
    with pytest.raises(RuntimeError, match="frame read failed"):
        camera.get(timeout=1)
    camera.close()
    assert capture.release_count == 1


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1])
def test_invalid_camera_timeouts_are_rejected(controlled_camera, timeout):
    camera, _ = controlled_camera
    with pytest.raises(ValueError, match="timeout"):
        camera.get(timeout=timeout)
    with pytest.raises(ValueError, match="timeout"):
        camera.close(timeout=timeout)


def test_constructor_failure_releases_opened_device(monkeypatch):
    released = []

    def fail_setting(*_):
        raise RuntimeError("device configuration failed")

    device = SimpleNamespace(isOpened=lambda: True, set=fail_setting,
                             release=lambda: released.append(True))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=lambda _: device, CAP_PROP_BUFFERSIZE=1,
    ))
    with pytest.raises(RuntimeError, match="configuration failed"):
        LatestFrameCamera()
    assert released == [True]


def test_real_worker_keeps_one_latest_frame_and_counts_overwrites(monkeypatch):
    reads = Queue()
    released = []
    device = SimpleNamespace(
        isOpened=lambda: True, set=lambda *_: True,
        read=lambda: reads.get(timeout=5), release=lambda: released.append(True),
    )
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        VideoCapture=lambda _: device, CAP_PROP_BUFFERSIZE=1,
    ))
    camera = LatestFrameCamera()
    camera.start()
    try:
        for value in range(3):
            reads.put((True, np.full((2, 2, 3), value, dtype=np.uint8)))
            with camera._condition:
                assert camera._condition.wait_for(
                    lambda expected=value + 1: camera.captured_frames == expected, 1
                )
        assert camera.snapshot() == (3, 2, 1)
        frame = camera.get(timeout=0)
        assert frame.frame_id == 2 and np.all(frame.image == 2)
        assert frame.source_timestamp_ns > 0
        assert camera.snapshot() == (3, 2, 0)
    finally:
        reads.put((False, None))
        camera.close(timeout=1)
    assert released == [True]
