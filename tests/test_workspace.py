from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("rles_workspace_validator", ROOT / "packaging" / "workspace.py")
assert _SPEC is not None and _SPEC.loader is not None
_WORKSPACE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKSPACE)
validate_tree = _WORKSPACE.validate_tree
pytestmark = pytest.mark.unit


@pytest.mark.dependency_resolver
def test_all_packages_and_required_workspace_edges_are_declared() -> None:
    assert validate_tree(ROOT, require_all=True) == ()


def test_workspace_validator_rejects_a_missing_required_edge(tmp_path: Path) -> None:
    source = ROOT / "packages"
    for package in source.iterdir():
        target = tmp_path / "packages" / package.name
        target.mkdir(parents=True)
        (target / "pyproject.toml").write_bytes((package / "pyproject.toml").read_bytes())
    data_meta = tmp_path / "packages" / "data" / "pyproject.toml"
    data_meta.write_text(
        data_meta.read_text(encoding="utf-8").replace(
            'dependencies = ["rles-contracts>=0.4.0.dev0,<0.5.0"]', "dependencies = []"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data: required workspace dependencies missing"):
        validate_tree(tmp_path, require_all=True)
