from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


_GENERATOR = Path(__file__).resolve().parents[1] / "qa" / "source_manifest.py"


def _generate(root: Path, output: Path) -> dict:
    subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(root), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_source_identity_tracks_workspace_inputs_and_ignores_generated_reports(tmp_path: Path) -> None:
    (tmp_path / "packages" / "demo").mkdir(parents=True)
    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "packages" / "demo" / ".tmp").mkdir()
    (tmp_path / "packages" / "demo" / ".tmp" / "generated.py").write_text(
        "VALUE = 'temporary build output'\n", encoding="utf-8",
    )
    (tmp_path / "packages" / "demo" / "dist-build" / "generated.py").parent.mkdir()
    (tmp_path / "packages" / "demo" / "dist-build" / "generated.py").write_text(
        "VALUE = 'temporary distribution output'\n", encoding="utf-8",
    )
    (tmp_path / "packages" / "demo" / ".venv-runtime" / "Scripts").mkdir(parents=True)
    (tmp_path / "packages" / "demo" / ".venv-runtime" / "Scripts" / "python.exe").write_bytes(
        b"generated interpreter placeholder",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text("def test_demo(): pass\n", encoding="utf-8")
    (tmp_path / "qa" / "reports").mkdir(parents=True)
    (tmp_path / "qa" / "research_trial_inventory.py").write_text(
        "VALUE = 'trial inventory'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "train_signal_baselines.py").write_text(
        "VALUE = 'signal baseline runner'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage58_train_coverage.py").write_text(
        "VALUE = 'train coverage'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage59_train_diagnostics.py").write_text(
        "VALUE = 'stage59 analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_stage59_train_diagnostics.py").write_text(
        "VALUE = 'stage59 preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_stage60_fixed_regimes.py").write_text(
        "VALUE = 'stage60 fixed regime analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_stage60_fixed_regimes.py").write_text(
        "VALUE = 'stage60 fixed regime preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_m33_action_value.py").write_text(
        "VALUE = 'M3.3 action value analysis'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "preregister_m33_action_value.py").write_text(
        "VALUE = 'M3.3 action value preregistration'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "m13_stage_contribution_20261017.py").write_text(
        "VALUE = 'M1.3 stage contribution runner'\n", encoding="utf-8",
    )
    (tmp_path / "qa" / "analyze_m13_stage_contribution.py").write_text(
        "VALUE = 'M1.3 stage contribution independent analysis'\n", encoding="utf-8",
    )
    manifest_path = tmp_path / "qa" / "reports" / "source_manifest_current.json"

    first = _generate(tmp_path, manifest_path)
    second = _generate(tmp_path, manifest_path)
    assert first["source_identity_sha256"] == second["source_identity_sha256"]
    assert [item["path"] for item in first["files"]] == [
        "packages/demo/module.py", "qa/analyze_m13_stage_contribution.py",
        "qa/analyze_m33_action_value.py",
        "qa/analyze_stage58_train_coverage.py",
        "qa/analyze_stage59_train_diagnostics.py",
        "qa/analyze_stage60_fixed_regimes.py",
        "qa/m13_stage_contribution_20261017.py",
        "qa/preregister_m33_action_value.py",
        "qa/preregister_stage59_train_diagnostics.py",
        "qa/preregister_stage60_fixed_regimes.py",
        "qa/research_trial_inventory.py", "qa/train_signal_baselines.py",
        "tests/test_demo.py",
    ]

    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    changed = _generate(tmp_path, manifest_path)
    assert changed["source_identity_sha256"] != first["source_identity_sha256"]


def test_workspace_identity_does_not_report_the_manifest_as_dirty(tmp_path: Path) -> None:
    (tmp_path / "packages" / "demo").mkdir(parents=True)
    (tmp_path / "packages" / "demo" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "qa" / "reports" / "source_manifest_current.json"
    manifest_path.parent.mkdir(parents=True)
    subprocess.run(["git", "init", str(tmp_path)], check=True,
                   capture_output=True, text=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "tests@example.invalid"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "packages/demo/module.py"],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "initial"],
                   check=True, capture_output=True, text=True)

    manifest = _generate(tmp_path, manifest_path)

    assert manifest["workspace_git"]["tracked_worktree_clean"] is True
    assert manifest["workspace_git"]["status_entries"] == []
    subprocess.run(
        [sys.executable, str(_GENERATOR), "--root", str(tmp_path),
         "--output", str(manifest_path), "--verify"],
        check=True, capture_output=True, text=True,
    )
