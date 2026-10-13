from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rles_dependency_audit", ROOT / "packaging" / "dependency_audit.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)

pytestmark = pytest.mark.dependency_resolver


@pytest.mark.parametrize(
    ("filename", "expected_torch"),
    (
        ("cpu.lock", "2.14.1+cpu"),
        ("cuda.lock", "2.14.1+cu130"),
        ("cuda-linux.lock", "2.14.1+cu130"),
    ),
)
def test_runtime_lock_is_hash_complete_and_device_specific(
    filename: str, expected_torch: str
) -> None:
    pins = _AUDIT.read_hash_lock(ROOT / "requirements" / filename)
    assert pins["torch"][0] == expected_torch
    assert pins["gymnasium"][0] == "1.3.0"
    assert len(pins) >= 46
    assert all(hashes for _, hashes in pins.values())
