from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile


_GENERATOR = Path(__file__).resolve().parents[1] / "qa" / "wheel_manifest.py"


def _run(root: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(_GENERATOR),
            "--root", str(root),
            "--wheels", str(root / "wheels"),
            "--source-manifest", str(root / "source.json"),
            "--output", str(root / "report.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, json.loads((root / "report.json").read_text(encoding="utf-8"))


def test_wheel_manifest_matches_built_sources_and_detects_drift(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "demo"
    source = package / "src" / "demo" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    package.mkdir(exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "demo-pkg"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "demo_pkg-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("demo/__init__.py", source.read_bytes())
        archive.writestr(
            "demo_pkg-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0.0\n",
        )
    (tmp_path / "source.json").write_text(
        json.dumps({"source_identity_sha256": "a" * 64}), encoding="utf-8"
    )

    result, report = _run(tmp_path)
    assert result.returncode == 0
    assert report["source_matches_all_wheels"] is True
    assert report["wheels"][0]["source_file_count"] == 1

    source.write_text("VALUE = 2\n", encoding="utf-8")
    result, report = _run(tmp_path)
    assert result.returncode == 1
    assert report["source_matches_all_wheels"] is False
    assert report["wheels"][0]["content_mismatch"] == ["demo/__init__.py"]
