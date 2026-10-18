"""Exact owner lifetime and named owned Job checks; never terminate a PID."""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
import importlib
import os

from .errors import ComputeError
from .windows_job import _Accounting, _ExtendedLimits, _checked, _kernel


def _machine() -> str:
    winreg = importlib.import_module("winreg")

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
        error = getattr(ct, "get_last_error")()
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
        error = getattr(ct, "get_last_error")()
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


def _job_process_ids(handle) -> tuple[int, ...]:
    """Return current PIDs from JOB_OBJECT_BASIC_PROCESS_ID_LIST, failing closed if truncated."""
    capacity = 4096
    buffer = ct.create_string_buffer(8 + capacity * ct.sizeof(ct.c_size_t))
    _checked(_kernel().QueryInformationJobObject(
        handle, 3, buffer, ct.sizeof(buffer), None,
    ), "QueryInformationJobObject process IDs")
    count = int(ct.c_uint32.from_buffer(buffer, 4).value)
    if count > capacity:
        raise ComputeError("supervised Job process ID list is truncated")
    values = ct.cast(ct.byref(buffer, 8), ct.POINTER(ct.c_size_t * capacity)).contents
    return tuple(int(values[index]) for index in range(count))


class _ProcessEntry32W(ct.Structure):
    _fields_ = [
        ("size", wt.DWORD), ("usage", wt.DWORD), ("process_id", wt.DWORD),
        ("heap_id", ct.c_size_t), ("module_id", wt.DWORD), ("threads", wt.DWORD),
        ("parent_process_id", wt.DWORD), ("base_priority", ct.c_long),
        ("flags", wt.DWORD), ("executable", wt.WCHAR * 260),
    ]


def _process_parent_map() -> dict[int, int]:
    """Read one native process snapshot for conservative Job ancestry checks."""
    kernel = _kernel()
    kernel.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    kernel.Process32FirstW.argtypes = [wt.HANDLE, ct.POINTER(_ProcessEntry32W)]
    kernel.Process32FirstW.restype = wt.BOOL
    kernel.Process32NextW.argtypes = [wt.HANDLE, ct.POINTER(_ProcessEntry32W)]
    kernel.Process32NextW.restype = wt.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == wt.HANDLE(-1).value:
        raise ComputeError(f"cannot snapshot Windows process ancestry (error {getattr(ct, 'get_last_error')()})")
    try:
        entry = _ProcessEntry32W()
        entry.size = ct.sizeof(entry)
        if not kernel.Process32FirstW(snapshot, ct.byref(entry)):
            raise ComputeError(f"cannot read Windows process ancestry (error {getattr(ct, 'get_last_error')()})")
        parents: dict[int, int] = {}
        while True:
            parents[int(entry.process_id)] = int(entry.parent_process_id)
            if kernel.Process32NextW(snapshot, ct.byref(entry)):
                continue
            error = getattr(ct, "get_last_error")()
            if error != 18:  # ERROR_NO_MORE_FILES
                raise ComputeError(f"cannot finish Windows process ancestry (error {error})")
            return parents
    finally:
        kernel.CloseHandle(snapshot)


def _owner_ancestors(owner_pid: int, parents: dict[int, int]) -> tuple[int, ...]:
    ancestors: list[int] = []
    seen = {owner_pid}
    current = owner_pid
    for _ in range(1024):
        parent = parents.get(current)
        if parent is None or parent == 0:
            return tuple(ancestors)
        if parent in seen:
            raise ComputeError("Windows process ancestry contains a cycle")
        ancestors.append(parent)
        seen.add(parent)
        current = parent
    raise ComputeError("Windows process ancestry exceeds the supported depth")


def require_owned_job(name: str, memory_bytes: int, owner_pid: int) -> dict:
    """Prove this coordinator is in its exact named process-tree Job."""
    if (not isinstance(name, str) or not name.startswith("Local\\RLES-Supervisor-")
            or len(name) > 240 or type(memory_bytes) is not int or memory_bytes <= 0
            or type(owner_pid) is not int or owner_pid != os.getpid()):
        raise ComputeError("invalid supervised application Job identity")
    kernel = _kernel()
    handle = kernel.OpenJobObjectW(0x4, False, name)
    if not handle:
        raise ComputeError(f"cannot inspect supervised application Job (Windows error {getattr(ct, 'get_last_error')()})")
    process = kernel.OpenProcess(0x1000 | 0x100000, False, owner_pid)
    if not process:
        error = getattr(ct, 'get_last_error')()
        kernel.CloseHandle(handle)
        raise ComputeError(f"Windows OpenProcess supervised owner failed (Windows error {error})")
    try:
        limits, accounting = _ExtendedLimits(), _Accounting()
        _checked(kernel.QueryInformationJobObject(handle, 9, ct.byref(limits), ct.sizeof(limits), None),
                 "QueryInformationJobObject application recovery limits")
        required = 0x100 | 0x200 | 0x2000
        if (limits.basic.flags & required != required or limits.basic.flags & (0x800 | 0x1000)
                or limits.process_memory != memory_bytes or limits.job_memory != memory_bytes):
            raise ComputeError("supervised application Job limits changed")
        _checked(kernel.QueryInformationJobObject(handle, 1, ct.byref(accounting),
                                                 ct.sizeof(accounting), None),
                 "QueryInformationJobObject application recovery accounting")
        member = wt.BOOL()
        _checked(kernel.IsProcessInJob(process, handle, ct.byref(member)), "IsProcessInJob")
        if not member.value or not accounting.active_processes:
            raise ComputeError("supervised owner is not active in its exact Job")
        process_ids = _job_process_ids(handle)
        ancestors = _owner_ancestors(owner_pid, _process_parent_map())
        return {"job_name": name, "state": "owner_active", "owner_pid": owner_pid,
                "active_processes": int(accounting.active_processes),
                "process_ids": process_ids, "owner_ancestor_pids": ancestors,
                "memory_bytes": memory_bytes}
    finally:
        kernel.CloseHandle(process)
        kernel.CloseHandle(handle)
