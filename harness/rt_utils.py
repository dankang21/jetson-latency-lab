# SPDX-License-Identifier: Apache-2.0
"""
Real-time primitives for the latency harness.

Everything here is a thin ctypes/os wrapper over Linux RT facilities. The point
is to remove the measurement harness itself as a source of jitter:

  * absolute clock_nanosleep   -> periodic release with no accumulated drift
  * SCHED_FIFO + CPU affinity   -> the loop is not preempted by SCHED_OTHER work
  * mlockall                     -> no page faults on the hot path

These calls require CAP_SYS_NICE / CAP_IPC_LOCK, i.e. run under sudo.
"""

import ctypes
import ctypes.util
import os
import time

CLOCK_MONOTONIC = 1
TIMER_ABSTIME = 1
MCL_CURRENT = 1
MCL_FUTURE = 2
EINTR = 4


class _timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.clock_nanosleep.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_timespec),
    ctypes.POINTER(_timespec),
]
_libc.mlockall.argtypes = [ctypes.c_int]


def now_ns() -> int:
    """Monotonic clock in nanoseconds (same clock the sleep arms against)."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def sleep_until_ns(deadline_ns: int) -> None:
    """
    Sleep until an absolute CLOCK_MONOTONIC instant.

    Using TIMER_ABSTIME against a fixed schedule (next = start + n*period)
    means a late wake-up in cycle N does not push cycle N+1 -- error does not
    accumulate, unlike relative time.sleep(). clock_nanosleep returns the errno
    directly (positive) rather than via -1/errno; on EINTR we just re-arm to the
    same absolute deadline (the 4th arg is unused for ABSTIME).
    """
    ts = _timespec(tv_sec=deadline_ns // 1_000_000_000,
                   tv_nsec=deadline_ns % 1_000_000_000)
    while True:
        ret = _libc.clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME,
                                    ctypes.byref(ts), None)
        if ret == 0:
            return
        if ret == EINTR:
            continue
        raise OSError(ret, os.strerror(ret))


def set_realtime(priority: int = 80, cpu: int | None = None) -> None:
    """Pin to a CPU (optional) and switch the calling thread to SCHED_FIFO."""
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    param = os.sched_param(priority)
    os.sched_setscheduler(0, os.SCHED_FIFO, param)


def lock_memory() -> None:
    """mlockall(MCL_CURRENT|MCL_FUTURE): no major faults on the hot path."""
    if _libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def try_apply_rt(priority: int, cpu: int | None, lock_mem: bool) -> list[str]:
    """
    Best-effort apply. Returns a list of human-readable warnings for anything
    that failed (e.g. not running as root) so the run is still recorded but the
    metadata makes the conditions explicit.
    """
    warnings: list[str] = []
    try:
        set_realtime(priority, cpu)
    except PermissionError:
        warnings.append("SCHED_FIFO/affinity denied (need sudo) -> ran SCHED_OTHER")
    except OSError as e:
        warnings.append(f"set_realtime failed: {e}")
    if lock_mem:
        try:
            lock_memory()
        except OSError as e:
            warnings.append(f"mlockall failed: {e}")
    return warnings
