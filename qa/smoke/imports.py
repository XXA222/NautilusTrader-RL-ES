"""Import/version smoke for the merged RLES workspace on CPython 3.14.7."""

from __future__ import annotations

import importlib
from importlib.metadata import version
import sys

import torch

assert sys.implementation.name == "cpython"
assert sys.version_info[:3] == (3, 14, 7), sys.version

expected = {
    "nautilus-trader": "2.0.0rc6",
    "stable-baselines3": "2.9.0",
    "torch": "2.14.1",
    "evotorch": "0.6.1",
    "gymnasium": "1.3.0",
    "pyarrow": "25.0.1",
}
actual_versions = {}
for distribution, wanted in expected.items():
    actual = version(distribution)
    # CUDA wheels have a PEP 440 local build label, for example
    # ``2.14.1+cu130``. The CPU lock uses ``2.14.1+cpu``. Both share the
    # pinned upstream version; accelerator capability is checked separately.
    upstream = actual.split("+", 1)[0]
    assert upstream == wanted, f"{distribution}: expected {wanted}, got {actual}"
    actual_versions[distribution] = actual

modules = (
    "rles_contracts",
    "rles_data",
    "rles_nautilus_adapter",
    "rles_execution",
    "rles_risk",
    "rles_gym_bridge",
    "rles_rl_sb3",
    "rles_es_evotorch",
    "rles_orchestrator",
    "rles_compute_runtime",
    "rles_research",
)
for module in modules:
    importlib.import_module(module)

print(f"PYTHON={sys.version.split()[0]}")
print("VERSIONS=" + ",".join(f"{name}=={value}" for name, value in actual_versions.items()))
print(f"TORCH_CUDA={torch.version.cuda}; CUDA_AVAILABLE={torch.cuda.is_available()}")
print("IMPORT_OK=" + ",".join(modules))
