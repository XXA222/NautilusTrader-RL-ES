"""Separate, immutable hard/soft risk CONFIG contracts; no risk decisions here."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .schema import CONTRACT_VERSION, require_current_version
from .validation import require_decimal, require_hash, require_name


@dataclass(frozen=True, slots=True)
class HardRiskConfig:
    config_id: str
    max_abs_target: Decimal
    max_notional: Decimal
    maintenance_buffer: Decimal
    max_drawdown_fraction: Decimal
    signature_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.config_id, "config_id")
        require_hash(self.signature_hash, "signature_hash")
        for key in ("max_abs_target", "max_notional", "maintenance_buffer", "max_drawdown_fraction"):
            require_decimal(getattr(self, key), key)
        if self.max_abs_target <= 0 or self.max_notional <= 0:
            raise ValueError("hard exposure/notional limits must be positive")
        if not 0 <= self.maintenance_buffer < 1 or not 0 < self.max_drawdown_fraction <= 1:
            raise ValueError("hard risk fractions must be within bounds")


@dataclass(frozen=True, slots=True)
class SoftRiskParams:
    config_id: str
    exposure_multiplier: Decimal
    drawdown_penalty: Decimal
    turnover_penalty: Decimal
    signature_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.config_id, "config_id")
        require_hash(self.signature_hash, "signature_hash")
        for key in ("exposure_multiplier", "drawdown_penalty", "turnover_penalty"):
            require_decimal(getattr(self, key), key)
        if not 0 <= self.exposure_multiplier <= 1:
            raise ValueError("soft exposure multiplier cannot expand hard risk")
        if self.drawdown_penalty < 0 or self.turnover_penalty < 0:
            raise ValueError("soft penalties must be nonnegative")


SOFT_SEARCHABLE = frozenset(("soft.exposure_multiplier", "soft.drawdown_penalty", "soft.turnover_penalty"))


@dataclass(frozen=True, slots=True)
class E2SearchSpace:
    """Only soft parameter names can be evolved; no hard guard handle."""

    parameter_names: tuple[str, ...]
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.parameter_names) is not tuple or not self.parameter_names:
            raise ValueError("empty E2 search space")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("duplicate E2 parameter")
        if any(name not in SOFT_SEARCHABLE for name in self.parameter_names):
            raise ValueError("E2 may only search explicitly approved soft risk parameters")
