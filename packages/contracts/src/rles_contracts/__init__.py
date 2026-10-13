"""Nautilus-RLES independent L1 contracts, version 0.4.0.

Dependency-free import surface: no NautilusTrader, SB3, EvoTorch or torch imports.
"""

from .schema import CONTRACT_VERSION, PROTOCOL_VERSION
from .registry import ProviderDescriptor, ProviderRegistry

__version__ = "0.4.0"
__all__ = [
    "__version__",
    "CONTRACT_VERSION",
    "PROTOCOL_VERSION",
    "ProviderDescriptor",
    "ProviderRegistry",
]
