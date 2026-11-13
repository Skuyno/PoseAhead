"""Read the worker's command pipe without blocking Windows CRT stdin."""

from __future__ import annotations

import os
import select


def _pipe_chunks(stream, finished):
    fd = stream.fileno()
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        peek = kernel.PeekNamedPipe
        peek.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID,
            ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
        ]
        peek.restype = wintypes.BOOL
        get_type = kernel.GetFileType
        get_type.argtypes = [wintypes.HANDLE]
        get_type.restype = wintypes.DWORD
        handle = msvcrt.get_osfhandle(fd)
        # The private UI worker expects a pipe. DEVNULL/console invocations have
        # no UI command channel; do not introduce a blocking console reader.
        if get_type(handle) != 3:  # FILE_TYPE_PIPE
            return
        while not finished.is_set():
            available = wintypes.DWORD()
            if not peek(handle, None, 0, None, ctypes.byref(available), None):
                error = ctypes.get_last_error()
                if error in (109, 232):  # closed/broken pipe
                    return
                raise ctypes.WinError(error)
            if not available.value:
                finished.wait(0.05)
                continue
            # Sole reader: consume only bytes already available, never wait for
            # a line while holding a CRT lock. Native DLL initialization (and
            # inherited Git stdin) can otherwise deadlock against this thread.
            yield os.read(fd, min(available.value, 4096))
    else:
        while not finished.is_set():
            if select.select([fd], [], [], 0.05)[0]:
                chunk = os.read(fd, 4096)
                if not chunk:
                    return
                yield chunk


def listen_for_stop(stream, stop, finished):
    """Accept newline-delimited ``stop``; bound malformed/partial command memory."""
    pending = bytearray()
    oversized = False
    for chunk in _pipe_chunks(stream, finished):
        for byte in chunk:
            if byte == 10:
                if not oversized and pending.strip() == b"stop":
                    stop.set()
                    return
                pending.clear()
                oversized = False
            elif not oversized:
                if len(pending) < 256:
                    pending.append(byte)
                else:
                    pending.clear()
                    oversized = True
