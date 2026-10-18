"""Create and verify a reproducible identity manifest for the RLES workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


_SOURCE_ROOTS = ("packages", "tests", "configs", "ci", "packaging", "requirements", "qa/smoke")
_ROOT_FILES = (
    ".gitattributes",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "RLES_v0.4.0_总开发与集成基线.md",
    "qa/reconcile_native_equity_long_20261017.py",
    "qa/train_signal_baselines.py",
    "qa/analyze_stage58_train_coverage.py",
    "qa/analyze_stage59_train_diagnostics.py",
    "qa/preregister_stage59_train_diagnostics.py",
    "qa/analyze_stage60_fixed_regimes.py",
    "qa/preregister_stage60_fixed_regimes.py",
    "qa/analyze_m33_action_value.py",
    "qa/preregister_m33_action_value.py",
    "qa/m13_stage_contribution_20261017.py",
    "qa/analyze_m13_stage_contribution.py",
    "qa/research_trial_inventory.py",
)
_EXCLUDED_PARTS = {"__pycache__", "build", "dist", ".pytest_cache", ".mypy_cache", ".tmp"}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in _ROOT_FILES:
        path = root / relative
        if path.is_file() and not path.is_symlink():
            files.add(path)
    for relative in _SOURCE_ROOTS:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in directory.rglob("*"):
            generated_component = any(
                part in _EXCLUDED_PARTS or part.startswith((".venv", "dist"))
                for part in path.relative_to(root).parts
            )
            if (path.is_file() and not path.is_symlink()
                    and not generated_component
                    and path.suffix not in _EXCLUDED_SUFFIXES
                    and ".egg-info" not in path.as_posix()):
                files.add(path)
    workflow_dir = root / ".github" / "workflows"
    if workflow_dir.is_dir() and not workflow_dir.is_symlink():
        files.update(path for path in workflow_dir.rglob("*") if path.is_file() and not path.is_symlink())
    for relative in ("qa/source_manifest.py", "qa/wheel_manifest.py"):
        tool = root / relative
        if tool.is_file() and not tool.is_symlink():
            files.add(tool)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _upstream_identity(root: Path) -> dict[str, Any] | None:
    upstream = root / "nautilus_trader"
    if not upstream.is_dir():
        return None

    def git(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={upstream}", "-C", str(upstream), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    status = git("status", "--porcelain", "--untracked-files=all").decode("utf-8").splitlines()
    tracked_diff = git("diff", "HEAD", "--binary")
    return {
        "path": "nautilus_trader",
        "commit": commit,
        "tracked_worktree_clean": not any(not row.startswith("??") for row in status),
        "tracked_diff_sha256": _sha256(tracked_diff),
        "status_entries": status,
    }


def build_manifest(root: Path, *, generated_at_utc: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path.read_bytes()),
        }
        for path in _source_files(root)
    ]
    upstream = _upstream_identity(root)
    identity = {"files": files, "upstream_nautilus": upstream}
    identity_sha = _sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    installed = sorted(
        ({"name": dist.metadata.get("Name", ""), "version": dist.version} for dist in distributions()),
        key=lambda item: (item["name"].casefold(), item["version"]),
    )
    return {
        "schema": "rles.source-manifest.v1",
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "source_identity_sha256": identity_sha,
        "scope": {
            "included": [*_ROOT_FILES, *_SOURCE_ROOTS, ".github/workflows",
                         "qa/source_manifest.py", "qa/wheel_manifest.py"],
            "excluded_generated": [".tmp", ".venv*", "data", "dist*", "docs/audit reports", "qa/reports"],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_name": os.name,
            "installed_distributions": installed,
        },
        "upstream_nautilus": upstream,
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("qa/reports/source_manifest_current.json"))
    parser.add_argument("--verify", action="store_true", help="verify an existing manifest without rewriting it")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    fresh = build_manifest(root)
    if args.verify:
        if not output.is_file():
            print(f"manifest not found: {output}", file=sys.stderr)
            return 2
        previous = json.loads(output.read_text(encoding="utf-8"))
        matches = (
            previous.get("source_identity_sha256") == fresh["source_identity_sha256"]
            and previous.get("files") == fresh["files"]
            and previous.get("upstream_nautilus") == fresh["upstream_nautilus"]
        )
        if not matches:
            print("source manifest is stale", file=sys.stderr)
            return 1
        print(f"verified {len(fresh['files'])} files; source_identity_sha256={fresh['source_identity_sha256']}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}; files={len(fresh['files'])}; source_identity_sha256={fresh['source_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
