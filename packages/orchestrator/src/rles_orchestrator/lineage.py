"""L5-owned in-memory ancestry graph. Persistence is provided by L1 store."""
from __future__ import annotations

from dataclasses import dataclass

from .workflow import StageArtifact, lineage_hash


@dataclass(frozen=True)
class LineageEdge:
    parent_id: str
    child_id: str
    transition: str


@dataclass(frozen=True)
class LineageGraph:
    nodes: tuple[tuple[str, str, str, str], ...]
    edges: tuple[LineageEdge, ...]
    digest: str


def build_lineage(artifacts: tuple[StageArtifact, ...]) -> LineageGraph:
    digest = lineage_hash(artifacts)
    nodes = []
    edges = []
    observed = set()
    for idx, art in enumerate(artifacts):
        if art.artifact_id in observed:
            raise ValueError("ancestry graph contains repeated artifact identity")
        observed.add(art.artifact_id)
        nodes.append((art.stage, art.artifact_id, art.actor_hash, art.soft_risk_hash))
        if idx == 0:
            if art.parent_id is not None:
                raise ValueError("first graph node must have no parent")
        else:
            previous = artifacts[idx - 1]
            transition = previous.stage + "->" + art.stage
            if transition not in {"E1->R1", "R1->E2"}:
                raise ValueError("invalid first-version Hybrid transition")
            edges.append(LineageEdge(previous.artifact_id, art.artifact_id, transition))
    return LineageGraph(tuple(nodes), tuple(edges), digest)
