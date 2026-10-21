"""Run a complete application process tree inside a Windows memory-limited Job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from collections.abc import Sequence

from .errors import DeviceUnavailable, WorkerFailure
from .validation import valid_timeout
from .windows_job import WindowsWorkerJob


def run_windows_job(command: Sequence[str], *, memory_bytes: int, cwd: str | None = None,
                    timeout_s: float | None = None, job_name: str | None = None) -> dict:
    """Start an application suspended and enforce a commit-memory ceiling.

    The supervisor process itself remains outside the Job. The launched
    application and every descendant inherit the Job before the first
    application instruction runs. Job closure cleans up surviving descendants.
    """
    if os.name != "nt":
        raise DeviceUnavailable("whole-application Job supervision is Windows-only")
    if type(memory_bytes) is not int or memory_bytes <= 0:
        raise ValueError("positive process-tree memory limit required")
    if timeout_s is not None and not valid_timeout(timeout_s):
        raise ValueError("valid process timeout required")
    if job_name is None:
        job_name = f"Local\\RLES-Supervisor-{uuid.uuid4().hex}"
    args = tuple(command)
    if not args or any(not isinstance(arg, str) or not arg for arg in args):
        raise ValueError("nonempty process command required")
    command_hash = hashlib.sha256(
        json.dumps(args, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    job = WindowsWorkerJob(memory_bytes, name=job_name)
    inherited = {
        "RLES_SUPERVISION_JOB_NAME": os.environ.get("RLES_SUPERVISION_JOB_NAME"),
        "RLES_SUPERVISION_MEMORY_BYTES": os.environ.get("RLES_SUPERVISION_MEMORY_BYTES"),
    }
    os.environ["RLES_SUPERVISION_JOB_NAME"] = job_name
    os.environ["RLES_SUPERVISION_MEMORY_BYTES"] = str(memory_bytes)
    try:
        process_result = job.run_suspended(args, cwd=cwd, timeout_s=timeout_s)
    finally:
        try:
            cleanup = job.close()
        finally:
            for name, value in inherited.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    return {
        "schema": "rles.windows-supervised-process.v1",
        "command_sha256": command_hash,
        "memory_limit_bytes": memory_bytes,
        **process_result,
        "job_cleanup": cleanup,
    }


def _write_evidence(path_value: str, result: dict) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an application command under a Windows Job process-tree memory ceiling.",
    )
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--job-name", help="optional unique Local\\RLES- Job name for recovery")
    parser.add_argument("--cwd")
    parser.add_argument("--evidence", help="write a new JSON evidence file; existing files are never replaced")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("provide the application command after --")
    try:
        result = run_windows_job(command, memory_bytes=args.memory_bytes,
                                 cwd=args.cwd, timeout_s=args.timeout_seconds,
                                 job_name=args.job_name)
        if args.evidence:
            _write_evidence(args.evidence, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")), file=sys.stderr, flush=True)
        if result["timed_out"]:
            return 124
        return 0 if result["exit_code"] == 0 else 1
    except (DeviceUnavailable, ValueError, WorkerFailure, OSError) as exc:
        print(f"supervisor: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
