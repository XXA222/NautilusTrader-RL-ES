"""Strict injection via Python packaging entry points, no arbitrary fallback."""
from importlib.metadata import entry_points

_ALLOWED = frozenset({"nautilus_rles.es", "nautilus_rles.rl",
                      "nautilus_rles.compute", "nautilus_rles.engine"})


def resolve_provider(group: str, name: str):
    if group not in _ALLOWED or not isinstance(name, str) or not name.strip():
        raise ValueError("invalid frozen provider identity")
    providers = entry_points(group=group).select(name=name)
    if len(providers) != 1:
        raise LookupError(f"expected exactly one provider entry point: {group}/{name}")
    return providers[0].load()
