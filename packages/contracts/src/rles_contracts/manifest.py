"""Run identity (deterministic) and attempt identity (unique retry) with complete lineage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .schema import CONTRACT_VERSION, PROTOCOL_VERSION, require_current_version
from .serialization import canonical_hash
from .validation import require_hash


@dataclass(frozen=True, slots=True)
class RunFingerprints:
    """Changing any field produces a new content-addressed run ID."""

    dataset_hash: str
    feature_hash: str
    fold_hash: str
    config_hash: str
    code_hash: str
    dependency_hash: str
    seed_hash: str
    adapter_hash: str
    engine_hash: str
    compute_hash: str
    execution_hash: str
    risk_hash: str
    actor_hash: str
    numeric_profile_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        from dataclasses import fields
        for f in fields(self):
            if f.name != "schema_version":
                require_hash(getattr(self, f.name), f.name)

    @property
    def run_id(self) -> str:
        return canonical_hash({"protocol_version": PROTOCOL_VERSION, "fingerprints": self})


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    run_id: str
    attempt_id: str
    protocol_version: str
    fingerprints: RunFingerprints
    git_commit: str
    package_versions: tuple[tuple[str, str], ...]
    runtime: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    algorithm: tuple[tuple[str, str], ...]
    selection: tuple[tuple[str, str], ...]
    artifact_refs: tuple[tuple[str, str], ...]
    root_seed: int
    seed_tree_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("protocol version mismatch")
        if type(self.fingerprints) is not RunFingerprints or self.run_id != self.fingerprints.run_id:
            raise ValueError("run identity mismatch")
        require_hash(self.run_id, "run_id")
        try:
            parsed_attempt = uuid.UUID(self.attempt_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("attempt_id must be UUID") from exc
        if str(parsed_attempt) != self.attempt_id or parsed_attempt.version != 4:
            raise ValueError("attempt_id must be canonical lowercase UUIDv4")
        import re
        if type(self.git_commit) is not str or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.git_commit) is None:
            raise ValueError("git_commit must be 40/64 lowercase hex")
        if type(self.root_seed) is not int or not 0 <= self.root_seed < 1 << 64:
            raise ValueError("root_seed must be uint64")
        require_hash(self.seed_tree_hash, "seed_tree_hash")
        from .seed import SeedTree
        expected_seed_hash = SeedTree(self.root_seed).fingerprint()
        if self.seed_tree_hash != expected_seed_hash or self.fingerprints.seed_hash != expected_seed_hash:
            raise ValueError("seed tree/root seed must match run fingerprints")
        for field in ("package_versions", "runtime", "environment", "algorithm", "selection", "artifact_refs"):
            entries = getattr(self, field)
            if type(entries) is not tuple or any(type(e) is not tuple or len(e) != 2 or
                                                 any(type(v) is not str for v in e) for e in entries):
                raise TypeError(f"{field} must be tuple[(str,str),...]")
            keys = [k for k, _ in entries]
            if keys != sorted(set(keys)):
                raise ValueError(f"{field} keys must be sorted/unique")
            if any(not k or not v for k, v in entries):
                raise ValueError(f"{field} entries must have nonempty keys/values")
            if field != "artifact_refs" and not entries:
                raise ValueError(f"{field} must be nonempty for complete lineage")

    @classmethod
    def new_attempt(cls, fingerprints: RunFingerprints, *, git_commit: str,
                    package_versions: tuple[tuple[str, str], ...],
                    runtime: tuple[tuple[str, str], ...], environment: tuple[tuple[str, str], ...],
                    algorithm: tuple[tuple[str, str], ...], selection: tuple[tuple[str, str], ...],
                    artifact_refs: tuple[tuple[str, str], ...], root_seed: int,
                    seed_tree_hash: str) -> ExperimentManifest:
        return cls(fingerprints.run_id, str(uuid.uuid4()), PROTOCOL_VERSION, fingerprints,
                   git_commit, package_versions, runtime, environment, algorithm, selection,
                   artifact_refs, root_seed, seed_tree_hash)
