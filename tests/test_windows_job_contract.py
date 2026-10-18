from __future__ import annotations

import pytest

from rles_compute_runtime.errors import WorkerFailure
from rles_compute_runtime.windows_job import _resume_suspended_primary_thread


class _Kernel:
    def __init__(self, result: int):
        self.result = result

    def ResumeThread(self, _thread_handle) -> int:
        return self.result


def test_resume_suspended_primary_thread_requires_exactly_one_suspend() -> None:
    _resume_suspended_primary_thread(_Kernel(1), object())


@pytest.mark.parametrize("previous_suspend_count", [0, 2, 3])
def test_resume_suspended_primary_thread_rejects_unexpected_suspend_count(
    previous_suspend_count: int,
) -> None:
    with pytest.raises(WorkerFailure, match="expected exactly one"):
        _resume_suspended_primary_thread(_Kernel(previous_suspend_count), object())
