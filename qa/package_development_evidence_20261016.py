"""Save exact working source and small verification summaries without Git mutation."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import argparse
from zipfile import ZipFile, ZIP_DEFLATED

from rles_orchestrator.application import _file_hash, _verify_file, _write_verified


def package(root: Path, version: str = "v1") -> Path:
    if not version.isidentifier():
        raise ValueError("version must be a simple identifier")
    archive = root / ".artifacts/audit-development-20261016"
    evidence = root / "docs/audit_evidence/20261016"
    result = archive / "signal-baselines-v3/summary.json"
    _verify_file(result)
    shutil.copy2(result, evidence / "train_signal_baselines_summary.json")
    shutil.copy2(result.with_suffix(".json.sha256"),
                 evidence / "train_signal_baselines_summary.json.sha256")
    files = []
    for folder in (".github", "ci", "configs", "docs", "packaging", "packages",
                   "qa", "requirements", "tests"):
        for path in (root / folder).rglob("*"):
            if (path.is_file() and not {"build", "__pycache__", ".pytest_cache"}.intersection(path.parts)
                    and path.suffix not in {".pyc", ".pyo", ".log", ".sqlite"}):
                files.append(path)
    files += [root / name for name in (
        ".gitignore", ".gitattributes", "pyproject.toml", "uv.lock", "README.md", ".python-version")]
    files += list(evidence.glob("*.json")) + list(evidence.glob("*.sha256"))
    destination = archive / f"source-snapshot-{version}.zip"
    manifest = []
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as target:
        for path in sorted(set(files)):
            relative = path.relative_to(root).as_posix()
            target.write(path, relative)
            manifest.append({"path": relative, "sha256": _file_hash(path), "bytes": path.stat().st_size})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "--binary", "--no-ext-diff"], cwd=root,
                          capture_output=True, check=True).stdout
    diff_path = archive / f"working-tree-{version}.patch"
    with diff_path.open("xb") as target:
        target.write(diff)
    _write_verified(evidence / f"development_source_manifest_{version}.json", {
        "schema": "rles.development-source.v1", "git_head": head,
        "includes_preexisting_uncommitted_work": True,
        "zip": str(destination), "zip_sha256": _file_hash(destination),
        "patch": str(diff_path), "patch_sha256": _file_hash(diff_path), "files": manifest,
        "note": "source archive includes new untracked files; patch alone is insufficient",
    })
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    print(package(Path(__file__).resolve().parents[1], parser.parse_args().version))
