"""L1 ES checkpoint *storage envelope*, independent of EvoTorch's state codec.

Stores opaque bytes without deserializing them. An external G0-verified native
codec must prove full optimizer/distribution/population/RNG restoration before
these bytes may be used to continue evolution. No pickle load is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping
from types import MappingProxyType

_ALLOWED_COMPONENTS = frozenset({"distribution", "optimizer", "population",
                                 "rng_torch", "rng_numpy", "rng_python", "rng_evotorch",
                                 "batch_evaluator", "provider"})
_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_LIMIT_BYTES = 256 * 1024 * 1024


def _canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _write_sync(path: Path, data: bytes) -> None:
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


@dataclass(frozen=True)
class ESCheckpointEnvelope:
    algorithm: str
    attempt_id: str
    actor_signature: str
    dataset_hash: str
    config_hash: str
    generation: int
    evaluations: int
    data_cursor: int
    native_codec_id: str
    state_components: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if self.algorithm not in {"pgpe", "cmaes"} or not _ID.fullmatch(self.attempt_id):
            raise ValueError("invalid checkpoint identity")
        if not _ID.fullmatch(self.native_codec_id):
            raise ValueError("checkpoint native codec must be named and pinned")
        if any(not isinstance(x, str) or not _HEX.fullmatch(x) for x in (
            self.actor_signature, self.dataset_hash, self.config_hash
        )):
            raise ValueError("invalid checkpoint fingerprint")
        if any(type(x) is not int or x < 0 for x in (self.generation, self.evaluations, self.data_cursor)):
            raise ValueError("invalid generation/evaluation/cursor")
        if set(self.state_components) != _ALLOWED_COMPONENTS:
            raise ValueError("checkpoint lacks full distribution/optimizer/population/RNG components")
        if any(not isinstance(v, bytes) or not v or len(v) > _LIMIT_BYTES for v in self.state_components.values()):
            raise ValueError("invalid or oversized native checkpoint component")
        object.__setattr__(self, "state_components", MappingProxyType(dict(self.state_components)))


class ESCheckpointStore:
    """Atomic write-and-verify, fail closed on incomplete or corrupted checkpoints."""
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: ESCheckpointEnvelope) -> Path:
        if not isinstance(checkpoint, ESCheckpointEnvelope):
            raise TypeError("required ESCheckpointEnvelope")
        destination = self.root / checkpoint.attempt_id
        if destination.exists():
            raise FileExistsError("checkpoint immutable: existing attempt ID")
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            hashes = {}
            for name, blob in sorted(checkpoint.state_components.items()):
                _write_sync(staging / (name + ".bin"), blob)
                hashes[name] = sha256(blob).hexdigest()
            header = dict(version=1, algorithm=checkpoint.algorithm,
                          attempt_id=checkpoint.attempt_id,
                          actor_signature=checkpoint.actor_signature,
                          dataset_hash=checkpoint.dataset_hash,
                          config_hash=checkpoint.config_hash,
                          generation=checkpoint.generation,
                          evaluations=checkpoint.evaluations,
                          data_cursor=checkpoint.data_cursor,
                          native_codec_id=checkpoint.native_codec_id,
                          components=hashes)
            raw = _canonical(header)
            _write_sync(staging / "manifest.json", raw)
            _write_sync(staging / "COMPLETED", sha256(raw).hexdigest().encode("ascii"))
            os.rename(staging, destination)
            # Re-read verifies all contents before any cache entry can use it.
            self.load(checkpoint.attempt_id)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def load(self, attempt_id: str) -> ESCheckpointEnvelope:
        if not isinstance(attempt_id, str) or not _ID.fullmatch(attempt_id):
            raise ValueError("invalid attempt ID")
        directory = self.root / attempt_id
        if directory.is_symlink():
            raise RuntimeError("checkpoint directory must not be symbolic link")
        if not directory.is_dir():
            raise FileNotFoundError("checkpoint attempt does not exist")
        manifest = directory / "manifest.json"
        marker = directory / "COMPLETED"
        if marker.is_symlink() or manifest.is_symlink():
            raise RuntimeError("checkpoint metadata must not be symbolic links")
        if not marker.is_file() or not manifest.is_file():
            raise RuntimeError("incomplete checkpoint (missing COMPLETED marker or manifest)")
        data = manifest.read_bytes()
        if marker.read_text(encoding="ascii") != sha256(data).hexdigest():
            raise RuntimeError("checkpoint metadata integrity violation")
        payload = json.loads(data)
        expected_headers = {"version", "algorithm", "attempt_id", "actor_signature", "dataset_hash",
                            "config_hash", "generation", "evaluations", "data_cursor", "native_codec_id",
                            "components"}
        if not isinstance(payload, dict) or set(payload) != expected_headers:
            raise RuntimeError("checkpoint manifest has missing or unknown fields")
        if payload.get("version") != 1 or payload.get("attempt_id") != attempt_id:
            raise RuntimeError("checkpoint schema or identity mismatch")
        hashes = payload["components"]
        if not isinstance(hashes, dict) or set(hashes) != _ALLOWED_COMPONENTS:
            raise RuntimeError("checkpoint state coverage incorrect")
        expected_files = {"manifest.json", "COMPLETED"} | {n + ".bin" for n in _ALLOWED_COMPONENTS}
        if {p.name for p in directory.iterdir()} != expected_files:
            raise RuntimeError("unexpected checkpoint files")
        state = {}
        for name in sorted(_ALLOWED_COMPONENTS):
            path = directory / (name + ".bin")
            if path.is_symlink():
                raise RuntimeError("checkpoint component must not be symbolic link")
            if path.stat().st_size > _LIMIT_BYTES:
                raise RuntimeError("checkpoint component exceeds size limit")
            value = path.read_bytes()
            if sha256(value).hexdigest() != hashes[name]:
                raise RuntimeError(f"checkpoint component checksum mismatch: {name}")
            state[name] = value
        return ESCheckpointEnvelope(
            payload["algorithm"], payload["attempt_id"], payload["actor_signature"],
            payload["dataset_hash"], payload["config_hash"], payload["generation"],
            payload["evaluations"], payload["data_cursor"], payload["native_codec_id"], state,
        )
