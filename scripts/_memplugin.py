"""Temporary: each test's duration and the process's private bytes after it."""

import ctypes
import sys
import time


def _private_mb():
    if sys.platform != "win32":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    class Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_size_t) for name in (
            "cb", "PageFaultCount", "PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PrivateUsage / 2**20


_start = {}
_log = open("mem.txt", "w")


def pytest_runtest_setup(item):
    _start[item.nodeid] = time.time()


def pytest_runtest_teardown(item):
    import threading

    took = time.time() - _start.get(item.nodeid, time.time())
    print(f"[mem] {took:7.1f} s {_private_mb():9.0f} MB {threading.active_count():3d} threads "
          f"{item.nodeid}", file=_log, flush=True)
