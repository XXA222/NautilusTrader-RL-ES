"""L1-owned, read-only validator for the 11-package dependency DAG.

Never generates other teams' packages, modifies their pyproject files or claims
independent-wheel acceptance before every package exists.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

PACKAGES: dict[str, frozenset[str]] = {
    "contracts": frozenset(),
    "data": frozenset({"contracts"}),
    "nautilus_adapter": frozenset({"contracts"}),
    "execution": frozenset({"contracts"}),
    "risk": frozenset({"contracts"}),
    "gym_bridge": frozenset({"contracts"}),
    "rl_sb3": frozenset({"contracts", "gym_bridge"}),
    "es_evotorch": frozenset({"contracts"}),
    "compute_runtime": frozenset({"contracts"}),
    "orchestrator": frozenset({"contracts"}),
    "research": frozenset({"contracts"}),
}
DISTRIBUTIONS = {key: "rles-" + key.replace("_", "-") for key in PACKAGES}
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")


def topological_order(graph: dict[str, frozenset[str]]) -> tuple[str, ...]:
    if any(deps - graph.keys() for deps in graph.values()):
        raise ValueError("unknown dependency in workspace graph")
    result: list[str] = []
    pending = dict(graph)
    while pending:
        ready = sorted(key for key, deps in pending.items() if deps.issubset(result))
        if not ready:
            raise ValueError("cyclic dependency in workspace graph")
        for key in ready:
            result.append(key)
            del pending[key]
    return tuple(result)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_tree(root: Path, *, require_all: bool) -> tuple[str, ...]:
    topological_order(PACKAGES)
    pyprojects = {key: root / "packages" / key / "pyproject.toml" for key in PACKAGES}
    missing = tuple(key for key, path in pyprojects.items() if not path.is_file())
    if "contracts" in missing:
        raise ValueError("L1 contracts pyproject missing")
    if require_all and missing:
        raise ValueError(f"other workstream packages not supplied: {', '.join(missing)}")
    known = {_normalize(value): key for key, value in DISTRIBUTIONS.items()}
    present_graph: dict[str, frozenset[str]] = {}
    for key, path in pyprojects.items():
        if key in missing:
            continue
        with path.open("rb") as handle:
            meta = tomllib.load(handle)
        project = meta.get("project", {})
        expected = DISTRIBUTIONS[key]
        if _normalize(project.get("name", "")) != _normalize(expected):
            raise ValueError(f"{key}: project.name must be {expected}")
        deps = project.get("dependencies", [])
        if type(deps) is not list or any(type(x) is not str for x in deps):
            raise ValueError(f"{key}: project.dependencies must be list[str]")
        internal: set[str] = set()
        for item in deps:
            match = REQUIREMENT_NAME.match(item)
            if match is None:
                raise ValueError(f"{key}: unparseable dependency {item!r}")
            dependency = known.get(_normalize(match[1]))
            if dependency is not None:
                internal.add(dependency)
        unauthorized = internal - PACKAGES[key]
        if unauthorized:
            raise ValueError(f"{key}: undeclared cross-line dependency: {sorted(unauthorized)}")
        missing_edges = PACKAGES[key] - internal
        if missing_edges:
            raise ValueError(f"{key}: required workspace dependencies missing: {sorted(missing_edges)}")
        present_graph[key] = frozenset(internal)
    # The permitted graph is acyclic; validate the actual declared dependency graph as well.
    topological_order(present_graph)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    missing = validate_tree(args.root, require_all=args.require_all)
    print(f"required dependency edges present; {len(PACKAGES) - len(missing)}/{len(PACKAGES)} packages present")
    if missing:
        print("pending other owners: " + ", ".join(missing))


if __name__ == "__main__":
    main()
