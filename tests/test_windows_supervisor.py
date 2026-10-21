from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap

import pytest

from rles_compute_runtime.supervisor import run_windows_job


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")


def test_supervisor_caps_application_tree_and_preserves_checkpoint_on_limit(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    script = textwrap.dedent(f"""
        from pathlib import Path
        import time
        Path({str(checkpoint)!r}).write_text('checkpoint-before-limit', encoding='utf-8')
        allocation = bytearray(512 * 1024 * 1024)
        for offset in range(0, len(allocation), 4096):
            allocation[offset] = 1
        time.sleep(30)
    """)
    limit = 128 * 1024 * 1024
    result = run_windows_job((sys.executable, "-c", script), memory_bytes=limit, timeout_s=20)

    assert checkpoint.read_text(encoding="utf-8") == "checkpoint-before-limit"
    assert result["schema"] == "rles.windows-supervised-process.v1"
    assert result["exit_code"] != 0
    assert result["timed_out"] is False
    assert result["job"]["process_limit_bytes"] == limit
    assert result["job"]["tree_limit_bytes"] == limit
    assert result["job"]["peak_tree_commit_bytes"] <= limit
    assert result["job_cleanup"]["active_processes"] == 0


def test_supervisor_terminates_descendant_after_parent_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "descendant.pid"
    script = textwrap.dedent(f"""
        from pathlib import Path
        import subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(60)'],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        Path({str(pid_file)!r}).write_text(str(child.pid), encoding='ascii')
        time.sleep(0.2)
    """)
    result = run_windows_job((sys.executable, "-c", script),
                             memory_bytes=512 * 1024 * 1024, timeout_s=20)

    descendant_pid = int(pid_file.read_text(encoding="ascii"))
    assert result["exit_code"] == 0
    assert result["job"]["total_processes"] >= 2
    assert result["job_cleanup"]["active_processes"] == 0
    with pytest.raises(OSError):
        os.kill(descendant_pid, 0)


def test_supervisor_timeout_is_explicit_and_cleans_process_tree() -> None:
    result = run_windows_job((sys.executable, "-c", "import time; time.sleep(60)"),
                             memory_bytes=512 * 1024 * 1024, timeout_s=0.2)
    assert result["timed_out"] is True
    assert result["job_cleanup"]["active_processes"] == 0
