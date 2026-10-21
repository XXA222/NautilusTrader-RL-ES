"""Owned Windows worker process trees with commit-memory ceilings.

Limits apply after assignment, before the pool releases its factory startup
gate. The coordinator holds the sole non-inheritable Job handle, so its death
also terminates associated workers/descendants. This does not cap the parent.
"""
from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
from functools import lru_cache
import os
import subprocess
import time
from collections.abc import Sequence

from .errors import DeviceUnavailable, WorkerFailure
from .validation import valid_timeout


class _BasicLimits(ct.Structure):
    _fields_ = [("process_time", ct.c_longlong), ("job_time", ct.c_longlong),
                ("flags", wt.DWORD), ("min_working_set", ct.c_size_t),
                ("max_working_set", ct.c_size_t), ("active_limit", wt.DWORD),
                ("affinity", ct.c_size_t), ("priority", wt.DWORD), ("scheduling", wt.DWORD)]


class _IOCounters(ct.Structure):
    _fields_ = [(name, ct.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimits(ct.Structure):
    _fields_ = [("basic", _BasicLimits), ("io", _IOCounters),
                ("process_memory", ct.c_size_t), ("job_memory", ct.c_size_t),
                ("peak_process_memory", ct.c_size_t), ("peak_job_memory", ct.c_size_t)]


class _Accounting(ct.Structure):
    _fields_ = [("user_time", ct.c_longlong), ("kernel_time", ct.c_longlong),
                ("period_user_time", ct.c_longlong), ("period_kernel_time", ct.c_longlong),
                ("page_faults", wt.DWORD), ("total_processes", wt.DWORD),
                ("active_processes", wt.DWORD), ("terminated_processes", wt.DWORD)]


class _StartupInfo(ct.Structure):
    _fields_ = [("cb", wt.DWORD), ("reserved", wt.LPWSTR), ("desktop", wt.LPWSTR),
                ("title", wt.LPWSTR), ("x", wt.DWORD), ("y", wt.DWORD),
                ("x_size", wt.DWORD), ("y_size", wt.DWORD), ("x_count_chars", wt.DWORD),
                ("y_count_chars", wt.DWORD), ("fill_attribute", wt.DWORD),
                ("flags", wt.DWORD), ("show_window", wt.WORD), ("reserved_size", wt.WORD),
                ("reserved_bytes", ct.POINTER(ct.c_byte)), ("stdin", wt.HANDLE),
                ("stdout", wt.HANDLE), ("stderr", wt.HANDLE)]


class _ProcessInformation(ct.Structure):
    _fields_ = [("process", wt.HANDLE), ("thread", wt.HANDLE),
                ("process_id", wt.DWORD), ("thread_id", wt.DWORD)]


@lru_cache(maxsize=1)
def _kernel():
    if os.name != "nt":
        raise DeviceUnavailable("Windows Job memory enforcement is unavailable on this platform")
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ct.c_void_p, wt.LPCWSTR], wt.HANDLE),
        "OpenJobObjectW": ([wt.DWORD, wt.BOOL, wt.LPCWSTR], wt.HANDLE),
        "SetInformationJobObject": ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD], wt.BOOL),
        "QueryInformationJobObject": ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p], wt.BOOL),
        "OpenProcess": ([wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE),
        "IsProcessInJob": ([wt.HANDLE, wt.HANDLE, ct.POINTER(wt.BOOL)], wt.BOOL),
        "AssignProcessToJobObject": ([wt.HANDLE, wt.HANDLE], wt.BOOL),
        "TerminateJobObject": ([wt.HANDLE, wt.UINT], wt.BOOL),
        "CloseHandle": ([wt.HANDLE], wt.BOOL),
        "WaitForSingleObject": ([wt.HANDLE, wt.DWORD], wt.DWORD),
        "GetProcessTimes": ([wt.HANDLE, ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME),
                              ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME)], wt.BOOL),
        "ProcessIdToSessionId": ([wt.DWORD, ct.POINTER(wt.DWORD)], wt.BOOL),
        "CreateProcessW": ([wt.LPCWSTR, wt.LPWSTR, ct.c_void_p, ct.c_void_p,
                            wt.BOOL, wt.DWORD, ct.c_void_p, wt.LPCWSTR,
                            ct.POINTER(_StartupInfo), ct.POINTER(_ProcessInformation)], wt.BOOL),
        "ResumeThread": ([wt.HANDLE], wt.DWORD),
        "GetExitCodeProcess": ([wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
        "TerminateProcess": ([wt.HANDLE, wt.UINT], wt.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


def _checked(result, operation: str):
    if not result:
        raise WorkerFailure(f"Windows {operation} failed (winerror {getattr(ct, 'get_last_error')()})")
    return result


def _resume_suspended_primary_thread(kernel, thread_handle) -> None:
    """Resume only a thread known to have exactly one creation-time suspend."""
    previous_suspend_count = kernel.ResumeThread(thread_handle)
    if previous_suspend_count == 0xFFFFFFFF:
        raise WorkerFailure(
            f"Windows ResumeThread failed (winerror {getattr(ct, 'get_last_error')()})",
        )
    if previous_suspend_count != 1:
        raise WorkerFailure(
            "Windows suspended process had an unexpected primary-thread suspend count "
            f"({previous_suspend_count}); expected exactly one",
        )


class WindowsWorkerJob:
    """Single owned worker plus inherited descendants; no foreign Job lookup."""

    def __init__(self, memory_bytes: int, *, name: str | None = None):
        if type(memory_bytes) is not int or not 0 < memory_bytes <= ct.c_size_t(-1).value:
            raise ValueError("positive native-size memory ceiling required")
        self.memory_bytes = memory_bytes
        if name is not None and (not isinstance(name, str) or not name.startswith("Local\\RLES-")
                                 or len(name) > 240):
            raise ValueError("invalid owned Job name")
        self.name = name
        self._pid: int | None = None
        self._final: dict = {}
        kernel = _kernel()
        getattr(ct, "set_last_error")(0)
        self._handle = _checked(kernel.CreateJobObjectW(None, name), "CreateJobObjectW")
        if name is not None and getattr(ct, "get_last_error")() == 183:  # ERROR_ALREADY_EXISTS
            kernel.CloseHandle(self._handle)
            self._handle = None
            raise WorkerFailure("owned Job name already exists; refusing foreign object")
        limits = _ExtendedLimits()
        # PROCESS_MEMORY | JOB_MEMORY | KILL_ON_JOB_CLOSE. No breakaway flags.
        limits.basic.flags = 0x100 | 0x200 | 0x2000
        limits.process_memory = limits.job_memory = memory_bytes
        try:
            _checked(kernel.SetInformationJobObject(self._handle, 9, ct.byref(limits), ct.sizeof(limits)),
                     "SetInformationJobObject")
        except BaseException:
            kernel.CloseHandle(self._handle)
            self._handle = None
            raise

    def attach(self, pid: int) -> None:
        if (self._handle is None or self._pid is not None or type(pid) is not int
                or pid <= 0 or pid == os.getpid()):
            raise WorkerFailure("Job attachment requires a new owned worker, never its coordinator")
        kernel = _kernel()
        # Only call for a multiprocessing/subprocess child whose process handle
        # remains owned by the caller, preventing PID reuse during assignment.
        process = _checked(kernel.OpenProcess(0x100 | 0x1, False, pid), "OpenProcess")
        try:
            self.attach_handle(pid, process)
        finally:
            kernel.CloseHandle(process)

    def attach_handle(self, pid: int, process_handle) -> None:
        """Assign an already-owned process handle before its primary thread runs."""
        if (self._handle is None or self._pid is not None or type(pid) is not int
                or pid <= 0 or pid == os.getpid() or not process_handle):
            raise WorkerFailure("Job attachment requires a new owned process handle")
        _checked(_kernel().AssignProcessToJobObject(self._handle, process_handle),
                 "AssignProcessToJobObject")
        self._pid = pid

    def snapshot(self) -> dict:
        if self._handle is None:
            return dict(self._final)
        kernel = _kernel()
        limits, accounting = _ExtendedLimits(), _Accounting()
        _checked(kernel.QueryInformationJobObject(self._handle, 9, ct.byref(limits), ct.sizeof(limits), None),
                 "QueryInformationJobObject limits")
        _checked(kernel.QueryInformationJobObject(self._handle, 1, ct.byref(accounting), ct.sizeof(accounting), None),
                 "QueryInformationJobObject accounting")
        return {
            "enforcement": "windows_job_commit_memory", "pid": self._pid,
            **({"job_name": self.name} if self.name is not None else {}),
            "process_limit_bytes": int(limits.process_memory),
            "tree_limit_bytes": int(limits.job_memory),
            "peak_process_commit_bytes": int(limits.peak_process_memory),
            "peak_tree_commit_bytes": int(limits.peak_job_memory),
            "active_processes": int(accounting.active_processes),
            "total_processes": int(accounting.total_processes),
        }

    def close(self, *, timeout_s: float = 3) -> dict:
        if not valid_timeout(timeout_s):
            raise ValueError("valid Job shutdown timeout required")
        if self._handle is None:
            return dict(self._final)
        kernel = _kernel()
        state = self.snapshot()
        if state["active_processes"]:
            _checked(kernel.TerminateJobObject(self._handle, 1), "TerminateJobObject")
            deadline = time.monotonic() + timeout_s
            while True:
                state = self.snapshot()
                if state["active_processes"] == 0:
                    break
                if time.monotonic() >= deadline:
                    raise WorkerFailure("Job process tree survived shutdown; quota must remain reserved")
                time.sleep(0.01)
        self._final = state
        _checked(kernel.CloseHandle(self._handle), "CloseHandle Job")
        self._handle = None
        return dict(state)

    def run_suspended(self, command: Sequence[str], *, cwd: str | None = None,
                      timeout_s: float | None = None) -> dict:
        """Start a process suspended, assign it to this Job, then resume it.

        The child and all descendants are memory-limited before Python or another
        application instruction executes. The caller must close this Job after
        this method returns so any surviving descendants are terminated.
        """
        if self._handle is None or self._pid is not None:
            raise WorkerFailure("a fresh owned Job is required for supervised launch")
        args = tuple(command)
        if (not args or any(not isinstance(arg, str) or not arg for arg in args)
                or cwd is not None and (not isinstance(cwd, str) or not cwd)):
            raise ValueError("nonempty process command and optional working directory required")
        if timeout_s is not None and not valid_timeout(timeout_s):
            raise ValueError("valid process timeout required")

        kernel = _kernel()
        startup = _StartupInfo()
        startup.cb = ct.sizeof(startup)
        process_info = _ProcessInformation()
        command_line = ct.create_unicode_buffer(subprocess.list2cmdline(args))
        _checked(kernel.CreateProcessW(
            None, command_line, None, None, False, 0x4, None, cwd,
            ct.byref(startup), ct.byref(process_info),
        ), "CreateProcessW suspended")
        try:
            self.attach_handle(int(process_info.process_id), process_info.process)
            _resume_suspended_primary_thread(kernel, process_info.thread)
            timeout_ms = 0xFFFFFFFF if timeout_s is None else min(int(timeout_s * 1000), 0xFFFFFFFE)
            wait_result = kernel.WaitForSingleObject(process_info.process, timeout_ms)
            timed_out = wait_result == 258  # WAIT_TIMEOUT
            if timed_out:
                _checked(kernel.TerminateJobObject(self._handle, 1), "TerminateJobObject timeout")
                wait_result = kernel.WaitForSingleObject(process_info.process, 0xFFFFFFFF)
            if wait_result != 0:
                raise WorkerFailure(f"Windows supervised process wait failed (result {wait_result})")
            exit_code = wt.DWORD()
            _checked(kernel.GetExitCodeProcess(process_info.process, ct.byref(exit_code)),
                     "GetExitCodeProcess")
            return {"exit_code": int(exit_code.value), "timed_out": timed_out,
                    "job": self.snapshot()}
        except BaseException:
            if self._pid is None:
                kernel.TerminateProcess(process_info.process, 1)
                kernel.WaitForSingleObject(process_info.process, 0xFFFFFFFF)
            raise
        finally:
            kernel.CloseHandle(process_info.thread)
            kernel.CloseHandle(process_info.process)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
