"""Compare a built workspace wheelhouse with its frozen RLES source identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _forced_assets(package: Path, project: dict[str, Any]
                   ) -> tuple[dict[str, str], dict[str, bool]]:
    force_include = (project.get("tool", {}).get("hatch", {}).get("build", {})
                     .get("targets", {}).get("wheel", {}).get("force-include", {}))
    if not isinstance(force_include, dict):
        raise ValueError(f"{package}: wheel force-include must be a source-to-target mapping")
    assets: dict[str, str] = {}
    targets: dict[str, bool] = {}
    package_root = package.resolve()
    for source, target in force_include.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"{package}: invalid wheel force-include entry")
        source_relative = Path(source)
        target_relative = PurePosixPath(target.replace("\\", "/"))
        if (source_relative.is_absolute() or ".." in source_relative.parts
                or target_relative.is_absolute() or ".." in target_relative.parts):
            raise ValueError(f"{package}: wheel force-include path escapes the package")
        source_path = (package / source_relative).resolve()
        if not source_path.is_relative_to(package_root) or not source_path.exists():
            raise ValueError(f"{package}: missing or external force-include source {source}")
        destination_root = target_relative.as_posix()
        targets[destination_root] = source_path.is_dir()
        entries = ([source_path] if source_path.is_file() else
                   [path for path in source_path.rglob("*") if path.is_file()])
        for item in entries:
            if (item.is_symlink() or item.suffix in {".py", ".pyc", ".pyo"}
                    or "__pycache__" in item.relative_to(package_root).parts
                    or ".egg-info" in item.parts):
                continue
            relative = item.relative_to(source_path).as_posix() if source_path.is_dir() else ""
            destination = (target_relative / relative).as_posix() if relative else target_relative.as_posix()
            digest = _sha256(item.read_bytes())
            previous = assets.get(destination)
            if previous is not None and previous != digest:
                raise ValueError(f"{package}: conflicting wheel asset destination {destination}")
            assets[destination] = digest
    return assets, targets


def build_wheel_manifest(root: Path, wheels_dir: Path, source_identity: str) -> dict[str, Any]:
    root, wheels_dir = root.resolve(), wheels_dir.resolve()
    wheels = sorted(wheels_dir.glob("*.whl"))
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    packages = sorted((root / "packages").iterdir())
    expected = 0
    for package in packages:
        project_file = package / "pyproject.toml"
        source_root = package / "src"
        if not package.is_dir() or not project_file.is_file() or not source_root.is_dir():
            continue
        with project_file.open("rb") as handle:
            project_metadata = tomllib.load(handle)
        project = project_metadata.get("project", {})
        name, version = project.get("name"), project.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError(f"{project_file}: project name/version required")
        expected += 1
        source_assets, asset_targets = _forced_assets(package, project_metadata)
        matches: list[tuple[Path, dict[str, str], dict[str, str]]] = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                metadata_paths = [n for n in archive.namelist()
                                  if n.endswith(".dist-info/METADATA")]
                if len(metadata_paths) != 1:
                    continue
                metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_paths[0]))
                if _normalize(str(metadata.get("Name", ""))) != _normalize(name):
                    continue
                module_files = {
                    item.filename: _sha256(archive.read(item))
                    for item in archive.infolist()
                    if item.filename.endswith(".py")
                    and ".dist-info/" not in item.filename
                    and ".data/" not in item.filename
                }
                archive_names = set(archive.namelist())
                asset_names = {
                    name for name in archive_names
                    if any(
                        name.startswith(target.rstrip("/") + "/") if is_directory else name == target
                        for target, is_directory in asset_targets.items()
                    ) and not name.endswith(".py")
                    and ".dist-info/" not in name and ".data/" not in name
                }
                asset_files = {item: _sha256(archive.read(item)) for item in asset_names}
                matches.append((wheel, module_files, asset_files))
        if len(matches) != 1:
            errors.append(f"{name}: expected exactly one wheel, found {len(matches)}")
            continue

        wheel, built_files, built_assets = matches[0]
        source_files = {
            item.relative_to(source_root).as_posix(): _sha256(item.read_bytes())
            for item in source_root.rglob("*.py")
            if item.is_file() and not item.is_symlink()
        }
        missing = sorted(set(source_files) - set(built_files))
        extra = sorted(set(built_files) - set(source_files))
        mismatch = sorted(path for path in set(source_files) & set(built_files)
                          if source_files[path] != built_files[path])
        asset_missing = sorted(set(source_assets) - set(built_assets))
        asset_extra = sorted(set(built_assets) - set(source_assets))
        asset_mismatch = sorted(path for path in set(source_assets) & set(built_assets)
                                if source_assets[path] != built_assets[path])
        if missing or extra or mismatch or asset_missing or asset_extra or asset_mismatch:
            errors.append(f"{name}: missing={missing}, extra={extra}, content_mismatch={mismatch}")
            if asset_missing or asset_extra or asset_mismatch:
                errors.append(
                    f"{name}: asset_missing={asset_missing}, asset_extra={asset_extra}, "
                    f"asset_content_mismatch={asset_mismatch}"
                )
        records.append({
            "distribution": name,
            "version": version,
            "wheel": wheel.name,
            "wheel_sha256": _sha256(wheel.read_bytes()),
            "source_file_count": len(source_files),
            "wheel_python_file_count": len(built_files),
            "source_matches_wheel": not (missing or extra or mismatch),
            "missing": missing,
            "extra": extra,
            "content_mismatch": mismatch,
            "source_asset_count": len(source_assets),
            "wheel_asset_file_count": len(built_assets),
            "asset_missing": asset_missing,
            "asset_extra": asset_extra,
            "asset_content_mismatch": asset_mismatch,
        })
    if len(wheels) != expected:
        errors.append(f"expected {expected} workspace wheels, found {len(wheels)}")
    return {
        "schema": "rles.wheelhouse-manifest.v1",
        "source_identity_sha256": source_identity,
        "wheel_count": len(records),
        "source_matches_all_wheels": not errors and len(records) == expected,
        "wheels": records,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source_path = args.source_manifest if args.source_manifest.is_absolute() else root / args.source_manifest
    wheels_dir = args.wheels if args.wheels.is_absolute() else root / args.wheels
    output = args.output if args.output.is_absolute() else root / args.output
    source = json.loads(source_path.read_text(encoding="utf-8"))
    identity = source.get("source_identity_sha256")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("source manifest has no valid source_identity_sha256")
    report = build_wheel_manifest(root, wheels_dir, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}; wheels={report['wheel_count']}; all_match={report['source_matches_all_wheels']}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["source_matches_all_wheels"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
