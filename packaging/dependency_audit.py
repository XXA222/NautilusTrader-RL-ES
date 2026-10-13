"""L1 offline wheel/license auditor for G0, WITHOUT installing dependencies.

This is a *gate input verifier*, not a dependency resolver or Q0 certification.
Only a fully locked, hash-verified wheelhouse and reviewed SPDX decisions may
produce a draft SBOM and NOTICE. Outputs are external Q0 evidence, not bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

NAME = re.compile(r"[-_.]+")
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.!+~-]*)$")
HASH = re.compile(r"^--hash=sha256:([a-f0-9]{64})$")
SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")


def normalize(name: str) -> str:
    return NAME.sub("-", name).lower()


def read_hash_lock(path: Path) -> dict[str, tuple[str, frozenset[str]]]:
    """Accept only flat, exact ==pins with explicit SHA256 hashes; fail closed."""
    logical: list[str] = []
    pending = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.split("#", 1)[0].strip()
        if not raw:
            continue
        if raw.endswith("\\"):
            pending += raw[:-1] + " "
        else:
            logical.append(pending + raw)
            pending = ""
    if pending:
        raise ValueError("unterminated requirements continuation")
    locked: dict[str, tuple[str, frozenset[str]]] = {}
    for line in logical:
        tokens = line.split()
        match = PIN.fullmatch(tokens[0])
        if match is None or len(tokens) < 2:
            raise ValueError(f"not an exact hash-locked requirement: {line!r}")
        name, version = normalize(match[1]), match[2]
        hashes: set[str] = set()
        for token in tokens[1:]:
            item = HASH.fullmatch(token)
            if item is None:
                raise ValueError(f"unsupported lock token: {token!r}")
            hashes.add(item[1])
        if name in locked or not hashes:
            raise ValueError(f"duplicate/empty hashes for {name}")
        locked[name] = (version, frozenset(hashes))
    if not locked:
        raise ValueError("hash lock is empty")
    return locked


def read_approved_licenses(path: Path) -> dict[str, tuple[str, str]]:
    """SPDX must be reviewed and signed by a human; no fuzzy license guessing."""
    with path.open("rb") as handle:
        obj = tomllib.load(handle)
    if set(obj) != {"licenses"} or type(obj["licenses"]) is not dict:
        raise ValueError("expected only [licenses] table")
    approved: dict[str, tuple[str, str]] = {}
    for name, entry in obj["licenses"].items():
        norm = normalize(name)
        if name != norm or type(entry) is not dict or set(entry) != {"spdx", "reviewer"}:
            raise ValueError(f"{name}: expected normalized package with spdx and reviewer")
        spdx, reviewer = entry["spdx"], entry["reviewer"]
        if (type(spdx) is not str or SPDX.fullmatch(spdx) is None or
                type(reviewer) is not str or not reviewer.strip()):
            raise ValueError(f"{name}: license requires a manually reviewed SPDX ID")
        if norm in approved:
            raise ValueError(f"duplicated approved name {norm}")
        approved[norm] = (spdx, reviewer)
    return approved


def wheel_metadata(path: Path) -> tuple[str, str, str | None]:
    if not path.name.endswith(".whl"):
        raise ValueError("expected wheel, no sdists")
    with zipfile.ZipFile(path) as wheel:
        names = [info for info in wheel.infolist() if info.filename.endswith(".dist-info/METADATA")]
        if len(names) != 1 or names[0].file_size > 5_000_000:
            raise ValueError(f"{path.name}: expected one small wheel METADATA")
        meta = BytesParser(policy=default).parsebytes(wheel.read(names[0]))
    name, version = meta.get("Name"), meta.get("Version")
    license_expression = meta.get("License-Expression")
    if not name or not version:
        raise ValueError(f"{path.name}: missing wheel Name/Version")
    return normalize(str(name)), str(version), str(license_expression) if license_expression else None


def audit(wheelhouse: Path, lock: Path, license_decisions: Path) -> tuple[dict, str]:
    pins = read_hash_lock(lock)
    approvals = read_approved_licenses(license_decisions)
    if set(approvals) != set(pins):
        raise ValueError(f"license approval set differs from lock: {sorted(set(approvals) ^ set(pins))}")
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise ValueError("wheelhouse has no wheels")
    seen: set[str] = set()
    components: list[dict] = []
    notice = ["Nautilus-RLES Third-party Dependency NOTICE (reviewed draft)",
              "No license inferred automatically; check source distribution obligations.", ""]
    for wheel in wheels:
        name, version, claimed_license = wheel_metadata(wheel)
        if name not in pins:
            raise ValueError(f"unlocked wheel: {wheel.name}")
        if name in seen:
            raise ValueError(f"multiple wheels for same distribution in target wheelhouse: {name}")
        seen.add(name)
        pin_version, accepted = pins[name]
        digest = hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest()
        if version != pin_version or digest not in accepted:
            raise ValueError(f"wheel version/SHA256 mismatch: {wheel.name}")
        spdx, reviewer = approvals[name]
        if claimed_license and claimed_license != spdx:
            raise ValueError(f"{name}: wheel SPDX {claimed_license!r} differs from reviewed {spdx!r}")
        components.append({
            "type": "library", "name": name, "version": version,
            "hashes": [{"alg": "SHA-256", "content": digest}],
            "licenses": [{"license": {"id": spdx}}],
            "properties": [
                {"name": "rles:wheel_filename", "value": wheel.name},
                {"name": "rles:license_reviewer", "value": reviewer},
            ],
        })
        notice.append(f"{name} {version} | {spdx} | {wheel.name} | sha256:{digest} | reviewer:{reviewer}")
    if seen != set(pins):
        raise ValueError(f"wheelhouse missing locked distribution(s): {sorted(set(pins) - seen)}")
    sbom = {"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
            "components": sorted(components, key=lambda x: x["name"])}
    return sbom, "\n".join(notice) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True,
                        help="TOML: [licenses] then name = {spdx='MIT', reviewer='approved-by'}")
    parser.add_argument("--output", type=Path, required=True, help="Evidence directory, outside L1 zip")
    args = parser.parse_args()
    sbom, notice = audit(args.wheelhouse, args.lock, args.licenses)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sbom.cdx.json").write_text(
        json.dumps(sbom, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    (args.output / "NOTICE.txt").write_text(notice, encoding="utf-8")
    print(f"reviewed {len(sbom['components'])} hash-locked wheels; resolver/install smoke still required")


if __name__ == "__main__":
    main()
