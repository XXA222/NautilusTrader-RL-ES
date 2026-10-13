"""Versioned deterministic HMAC seed path derivation (no global RNG state)."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from .schema import CONTRACT_VERSION, require_current_version
from .validation import require_name

_DOMAIN = b"nautilus-rles.seedtree.v0.4.0\x00"


@dataclass(frozen=True, slots=True)
class SeedPath:
    segments: tuple[str, ...]
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.segments) is not tuple or not self.segments:
            raise ValueError("SeedPath requires a nonempty tuple")
        for segment in self.segments:
            require_name(segment, "seed segment")

    def extend(self, name: str) -> SeedPath:
        return SeedPath(self.segments + (name,))

    @classmethod
    def for_worker(cls, *, fold: str, stage: str, algorithm: str, env: int,
                   worker: int, stream: str) -> SeedPath:
        """Standard run-independent namespace: fold/stage/algo/env/worker/RNG."""
        require_name(fold, "fold")
        if stage not in ("E1", "R1", "E2"):
            raise ValueError("stage must be E1/R1/E2")
        if algorithm not in ("ppo", "sac", "td3", "ddpg", "pgpe", "cma_es"):
            raise ValueError("unknown algorithm seed namespace")
        if stream not in ("python", "numpy", "torch", "cuda", "environment", "es"):
            raise ValueError("unknown RNG stream")
        for label, index in (("env", env), ("worker", worker)):
            if type(index) is not int or not 0 <= index < 1 << 32:
                raise ValueError(f"{label} seed index must be uint32")
        return cls((f"fold:{fold}", f"stage:{stage}", f"algo:{algorithm}",
                    f"env:{env:08d}", f"worker:{worker:08d}", f"rng:{stream}"))


@dataclass(frozen=True, slots=True)
class SeedTree:
    root_seed: int
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.root_seed) is not int or not 0 <= self.root_seed < 1 << 64:
            raise ValueError("seed must be uint64, no bool/float")

    def derive_bytes(self, path: SeedPath) -> bytes:
        if type(path) is not SeedPath:
            raise TypeError("path must be SeedPath")
        payload = bytearray(_DOMAIN)
        for segment in path.segments:
            encoded = segment.encode("utf-8")
            payload.extend(len(encoded).to_bytes(4, "big"))
            payload.extend(encoded)
        return hmac.new(self.root_seed.to_bytes(8, "big"), payload, hashlib.sha256).digest()

    def derive_uint64(self, path: SeedPath) -> int:
        return int.from_bytes(self.derive_bytes(path)[:8], "big")

    def fingerprint(self) -> str:
        return hashlib.sha256(_DOMAIN + self.root_seed.to_bytes(8, "big")).hexdigest()
