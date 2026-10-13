"""Exact owner lifetime and named owned Job checks; never terminate a PID."""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
import os

from .errors import ComputeError
from .windows_job import _Accounting, _ExtendedLimits, _checked, _kernel


def _machine() -> str:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography",
                            access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
    except OSError as exc:
        raise ComputeError("cannot establish local machine identity") from exc
    if not isinstance(value, str) or not value:
        raise ComputeError("invalid local machine identity")
    return value


def _session(pid: int) -> int:
    value = wt.DWORD()
    _checked(_kernel().ProcessIdToSessionId(pid, ct.byref(value)), "ProcessIdToSessionId")
    return int(value.value)


def _created(handle) -> int:
    created, exited, kernel_time, user_time = (wt.FILETIME() for _ in range(4))
    _checked(_kernel().GetProcessTimes(handle, ct.byref(created), ct.byref(exited),
                                       ct.byref(kernel_time), ct.byref(user_time)), "GetProcessTimes")
    return int(created.dwHighDateTime) << 32 | int(created.dwLowDateTime)


def owner_identity() -> dict:
    kernel = _kernel()
    pid = os.getpid()
    handle = _checked(kernel.OpenProcess(0x1000 | 0x100000, False, pid), "OpenProcess owner")
    try:
        return {"pid": pid, "created": _created(handle), "machine": _machine(), "session": _session(pid)}
    finally:
        kernel.CloseHandle(handle)


def require_owner_exited(identity: dict) -> dict:
    if (not isinstance(identity, dict) or set(identity) != {"pid", "created", "machine", "session"}
            or type(identity["pid"]) is not int or identity["pid"] <= 0
            or type(identity["created"]) is not int or identity["created"] <= 0
            or type(identity["session"]) is not int or identity["session"] < 0
            or not isinstance(identity["machine"], str)):
        raise ComputeError("invalid recovery owner identity")
    if identity["machine"] != _machine() or identity["session"] != _session(os.getpid()):
        raise ComputeError("recovery requires the original machine and Windows session")
    kernel = _kernel()
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, identity["pid"])
    if not handle:
        error = ct.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: process no longer exists.
            return {"owner_state": "absent"}
        raise ComputeError(f"cannot establish owner exit (Windows error {error})")
    try:
        created = _created(handle)
        if created != identity["created"]:
            return {"owner_state": "original_exited_pid_reused"}
        state = kernel.WaitForSingleObject(handle, 0)
        if state == 0:
            return {"owner_state": "exited"}
        if state == 258:
            raise ComputeError("lease owner is still alive; recovery refused")
        raise ComputeError("owner lifetime status is unknown; recovery refused")
    finally:
        kernel.CloseHandle(handle)


def require_empty_job(name: str, memory_bytes: int) -> dict:
    kernel = _kernel()
    handle = kernel.OpenJobObjectW(0x4, False, name)  # JOB_OBJECT_QUERY only.
    if not handle:
        error = ct.get_last_error()
        if error == 2:  # ERROR_FILE_NOT_FOUND, after owner and kill-on-close Job destruction.
            return {"job_name": name, "state": "absent", "active_processes": 0}
        raise ComputeError(f"cannot inspect owned Job (Windows error {error})")
    try:
        limits, accounting = _ExtendedLimits(), _Accounting()
        _checked(kernel.QueryInformationJobObject(handle, 9, ct.byref(limits), ct.sizeof(limits), None),
                 "QueryInformationJobObject recovery limits")
        required = 0x100 | 0x200 | 0x2000
        if (limits.basic.flags & required != required or limits.basic.flags & (0x800 | 0x1000)
                or limits.process_memory != memory_bytes or limits.job_memory != memory_bytes):
            raise ComputeError("owned Job limits changed; recovery refused")
        _checked(kernel.QueryInformationJobObject(handle, 1, ct.byref(accounting), ct.sizeof(accounting), None),
                 "QueryInformationJobObject recovery accounting")
        if accounting.active_processes:
            raise ComputeError("owned Job still contains active processes; recovery refused")
        return {"job_name": name, "state": "empty", "active_processes": 0}
    finally:
        kernel.CloseHandle(handle)
