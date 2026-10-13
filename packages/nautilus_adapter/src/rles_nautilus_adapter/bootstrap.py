"""Official public API bootstrap. Native objects exist only in this adapter package."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from .submitter import NativePlanSubmitter
from .session import EngineBoundaryError


@dataclass(frozen=True, slots=True)
class NativeBootConfiguration:
    """Native values are created inside adapter code, never placed in L1 DTOs."""
    venue_kwargs: tuple[dict[str, Any], ...]
    instrument_builders: tuple[Callable[[], Any], ...]
    strategy_builder: Callable[[], Any]


class PublicBacktestBuilder:
    """Single-use builder: native execution models must not leak across folds."""
    def __init__(self, *, config: NativeBootConfiguration,
                 plan_submitter: NativePlanSubmitter | None = None) -> None:
        if not config.venue_kwargs or not config.instrument_builders or not callable(config.strategy_builder) or not all(callable(f) for f in config.instrument_builders):
            raise ValueError("complete venue, instrument and strategy factories required")
        # Freeze caller-owned containers. Native fee/fill/latency/margin model
        # objects can carry state; this builder can consume them only once.
        venues = []
        for value in config.venue_kwargs:
            venue = dict(value)
            for name in ("starting_balances", "modules"):
                if isinstance(venue.get(name), (list, tuple)):
                    venue[name] = tuple(venue[name])
            venues.append(MappingProxyType(venue))
        self._venues = tuple(venues)
        self._instrument_builders = tuple(config.instrument_builders)
        self._strategy_builder = config.strategy_builder
        self._submitter = plan_submitter
        self._consumed = False

    def __call__(self, episode: Any) -> Any:
        if self._consumed:
            raise EngineBoundaryError("native builder/models cannot be reused across episodes")
        self._consumed = True
        from nautilus_trader.backtest import BacktestEngine
        from nautilus_trader.config import BacktestEngineConfig
        from nautilus_trader.common import LoggerConfig, LogLevel
        # No reset or native object reuse: new Rust engine for every fold/attempt.
        # Historical Gym steps incrementally advance this streaming engine;
        # keep native output at ERROR so long market replays do not emit one
        # INFO banner per decision step. Errors still reach the CLI log.
        engine = BacktestEngine(BacktestEngineConfig(
            shutdown_on_error=True, bypass_logging=True, run_analysis=False,
            logging=LoggerConfig(
                stdout_level=LogLevel.ERROR, fileout_level=LogLevel.OFF,
                print_config=False,
            ),
        ))
        try:
            required = {"venue", "oms_type", "account_type", "starting_balances", "base_currency",
                        "margin_model", "fee_model", "fill_model", "latency_model",
                        "bar_execution", "liquidation_enabled"}
            for venue in self._venues:
                if required - set(venue):
                    raise ValueError(f"venue execution model incomplete: {sorted(required-set(venue))}")
                if any(venue[n] is None for n in ("margin_model", "fee_model", "fill_model", "latency_model")):
                    raise ValueError("native fee/margin/fill/latency models must be explicitly resolved")
                if venue["base_currency"] is None:
                    raise ValueError("native account base currency required for scoreable equity")
                if type(venue["bar_execution"]) is not bool or type(venue["liquidation_enabled"]) is not bool:
                    raise TypeError("venue bar/liquidation capabilities must be explicit bool")
                engine.add_venue(**venue)
            for builder in self._instrument_builders:
                engine.add_instrument(builder())
            strategy = self._strategy_builder()
            engine.add_strategy(strategy)
            if self._submitter is not None:
                self._submitter.bind(engine, strategy)
            return engine
        except BaseException:
            engine.dispose()
            raise
