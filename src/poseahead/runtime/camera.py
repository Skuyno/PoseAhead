"""Bounded-latency camera capture with a single latest-frame slot."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CameraFrame:
    frame_id: int
    source_timestamp_ns: int
    image: NDArray[np.uint8]


class LatestFrameCamera:
    """Capture on a background thread and discard stale, unconsumed frames."""

    def __init__(self, camera_index: int = 0) -> None:
        import cv2

        self._capture = cv2.VideoCapture(camera_index)
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"could not open camera {camera_index}")
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            self._capture.release()
            raise
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: CameraFrame | None = None
        self._error: str | None = None
        self._captured_frames = 0
        self._dropped_frames = 0
        self._started = False
        self._closed = False
        self._released = False
        self._release_error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="poseahead-camera", daemon=True)

    @property
    def captured_frames(self) -> int:
        with self._condition:
            return self._captured_frames

    @property
    def dropped_frames(self) -> int:
        with self._condition:
            return self._dropped_frames

    def snapshot(self) -> tuple[int, int, int]:
        with self._condition:
            return self._captured_frames, self._dropped_frames, int(self._latest is not None)

    def discard_pending_and_snapshot(self) -> tuple[int, int, int]:
        with self._condition:
            if self._latest is not None:
                self._dropped_frames += 1
            self._latest = None
            return self._captured_frames, self._dropped_frames, 0

    def start(self) -> None:
        with self._condition:
            if self._closed or self._started:
                raise RuntimeError("camera is closed or already started")
            try:
                self._thread.start()
            except Exception:
                self._closed = True
                self._stop.set()
                self._release_capture()
                raise
            self._started = True

    def get(self, timeout: float = 1.0) -> CameraFrame | None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("camera timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None and self._error is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._closed:
                return None
            if self._error is not None:
                raise RuntimeError(self._error)
            frame = self._latest
            self._latest = None
            return frame

    def close(self, timeout: float = 2.0) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("camera shutdown timeout must be finite and non-negative")
        with self._condition:
            self._closed = True
            self._stop.set()
            if self._latest is not None:
                self._dropped_frames += 1
            self._latest = None
            self._condition.notify_all()
            started = self._started
            if not started:
                self._release_capture()
        if started:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise RuntimeError("camera shutdown timed out; capture thread is still running")
        if self._release_error is not None:
            raise RuntimeError("camera release failed") from self._release_error

    def _release_capture(self) -> None:
        # Once started, only the capture thread may release the device: never race read().
        if not self._released:
            self._released = True
            try:
                self._capture.release()
            except Exception as error:
                self._release_error = error

    def __enter__(self) -> LatestFrameCamera:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                success, image = self._capture.read()
                timestamp_ns = time.perf_counter_ns()
                with self._condition:
                    if self._stop.is_set():
                        break
                    if not success:
                        raise RuntimeError("camera frame read failed")
                    if self._latest is not None:
                        self._dropped_frames += 1
                    frame_id = self._captured_frames
                    self._captured_frames += 1
                    self._latest = CameraFrame(frame_id, timestamp_ns, image)
                    self._condition.notify()
        except Exception as error:  # the consumer receives capture-thread failures
            with self._condition:
                if not self._stop.is_set():
                    self._error = f"camera capture stopped: {error}"
                self._condition.notify_all()
        finally:
            self._release_capture()
