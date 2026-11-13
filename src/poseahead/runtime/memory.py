"""Current-process resident memory, shared by live and recorded stability probes."""

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _windows_reader():
    """Load DLL functions and define their ABI once, not on every sample."""
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "paged_peak",
                "paged",
                "nonpaged_peak",
                "nonpaged",
                "pagefile",
                "pagefile_peak",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_process = kernel.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    read_memory = psapi.GetProcessMemoryInfo
    read_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    read_memory.restype = wintypes.BOOL

    def read():
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if read_memory(get_process(), ctypes.byref(counters), counters.cb):
            return counters.working / 1024**2
        return None

    return read


def working_set_mib():
    """Current resident process memory, or None on unsupported hosts (not peak RSS)."""
    if sys.platform == "win32":
        return _windows_reader()()
    if sys.platform.startswith("linux"):
        try:
            resident = int(Path("/proc/self/statm").read_text().split()[1])
            return resident * os.sysconf("SC_PAGE_SIZE") / 1024**2
        except (OSError, ValueError, IndexError):
            return None
    return None


def memory_summary(samples, *, minimum_duration_s=590, growth_limit_mib=128):
    valid = [(seconds, value) for seconds, value in samples if value is not None]
    result = {
        "samples_seconds_working_set_mib": samples,
        "scope": "current process working set",
        "growth_limit_mib": growth_limit_mib,
        "minimum_duration_s": minimum_duration_s,
        "growth_mib": None,
        "peak_mib": None,
        "status": "unavailable",
    }
    if not valid:
        return result
    growth = valid[-1][1] - valid[0][1]
    result.update(growth_mib=growth, peak_mib=max(value for _, value in valid))
    if valid[-1][0] - valid[0][0] < minimum_duration_s:
        result["status"] = "insufficient_duration"
    else:
        result["status"] = "passed" if growth <= growth_limit_mib else "failed"
    return result
