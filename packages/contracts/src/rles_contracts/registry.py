"""Lazy, strict Python entry-point discovery; no hard-coded provider imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .errors import ErrorCategory, RLESError
from .schema import Version
from .validation import require_name

ENTRY_POINT_GROUPS = {
    "engine": "nautilus_rles.engine",
    "rl": "nautilus_rles.rl",
    "es": "nautilus_rles.es",
    "compute": "nautilus_rles.compute",
}


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Installed entry-point contract shared by all implementation layers.

    The descriptor keeps dependency injection explicit: settings may contain
    already validated runtime objects (for example an environment factory),
    while implementation packages own their concrete construction rules.
    """

    kind: str
    contract_version: str
    factory: Callable[[Mapping[str, object]], object]

    def __post_init__(self) -> None:
        if self.kind not in ENTRY_POINT_GROUPS:
            raise ValueError("unknown provider kind")
        if not callable(self.factory):
            raise TypeError("provider factory must be callable")
        Version.parse(self.contract_version)

    def create(self, settings: Mapping[str, object]) -> object:
        if not isinstance(settings, Mapping):
            raise TypeError("provider settings must be a mapping")
        return self.factory(settings)


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    kind: str
    name: str
    distribution: str
    distribution_version: str
    contract_version: str


class ProviderRegistry:
    def __init__(self, *, contract_version: str = "0.4.0") -> None:
        self.contract_version = Version.parse(contract_version)

    def discover(self, kind: str) -> tuple[metadata.EntryPoint, ...]:
        if kind not in ENTRY_POINT_GROUPS:
            raise RLESError(ErrorCategory.CONFIG, "PROVIDER_KIND", "unknown provider kind")
        group = ENTRY_POINT_GROUPS[kind]
        candidates = tuple(metadata.entry_points(group=group))
        names = [ep.name for ep in candidates]
        if len(names) != len(set(names)):
            raise RLESError(ErrorCategory.PROVIDER, "PROVIDER_DUPLICATE",
                            f"ambiguous provider names in {group}")
        return tuple(sorted(candidates, key=lambda ep: ep.name))

    def load(self, kind: str, name: str) -> tuple[ProviderInfo, Any]:
        require_name(name, "provider name")
        found = [ep for ep in self.discover(kind) if ep.name == name]
        if not found:
            raise RLESError(ErrorCategory.PROVIDER, "PROVIDER_MISSING",
                            f"provider {kind}:{name} is not installed")
        ep = found[0]
        # Distribution metadata may be absent in user-constructed entry points:
        # never present it as an installed wheel with verified provenance.
        dist = ep.dist
        if dist is None:
            raise RLESError(ErrorCategory.SECURITY, "PROVIDER_PROVENANCE_MISSING",
                            f"provider {kind}:{name} has no distribution provenance")
        dist_name = dist.metadata.get("Name")
        if type(dist_name) is not str or not dist_name or type(dist.version) is not str:
            raise RLESError(ErrorCategory.SECURITY, "PROVIDER_PROVENANCE_INVALID",
                            f"provider {kind}:{name} has invalid wheel metadata")
        # Verify distribution attribution before executing the entry point.
        try:
            obj = ep.load()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RLESError(ErrorCategory.DEPENDENCY, "PROVIDER_IMPORT_FAILED",
                            f"provider {kind}:{name} dependencies unavailable") from exc
        except Exception as exc:
            raise RLESError(ErrorCategory.PROVIDER, "PROVIDER_LOAD_FAILED",
                            f"provider {kind}:{name} entry point failed to load") from exc
        if getattr(obj, "kind", None) != kind:
            raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_KIND_MISMATCH",
                            f"provider {kind}:{name} declares different kind")
        version = getattr(obj, "contract_version", None)
        if not isinstance(version, str):
            raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_VERSION_INVALID",
                            f"provider {kind}:{name} missing valid contract version")
        try:
            parsed = Version.parse(version)
        except (TypeError, ValueError) as exc:
            raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_VERSION_INVALID",
                            f"provider {kind}:{name} missing valid contract version") from exc
        if parsed.major != self.contract_version.major or parsed.minor != self.contract_version.minor:
            raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_VERSION_CONFLICT",
                            f"provider requires {version}; loader contracts {self.contract_version}")
        if not callable(getattr(obj, "create", None)):
            raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_FACTORY_INVALID",
                            f"provider {kind}:{name} missing create(settings) method")
        if not isinstance(obj, ProviderDescriptor):
            # Accept structurally compatible descriptors from separately
            # versioned distributions, while rejecting bare factory functions.
            try:
                from inspect import signature

                parameters = signature(obj.create).parameters
            except (TypeError, ValueError, AttributeError) as exc:
                raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_FACTORY_INVALID",
                                f"provider {kind}:{name} has an unverifiable factory") from exc
            if len(parameters) != 1:
                raise RLESError(ErrorCategory.CONTRACT, "PROVIDER_FACTORY_INVALID",
                                f"provider {kind}:{name} create must accept settings")
        info = ProviderInfo(kind, name, dist_name, dist.version, version)
        return info, obj
