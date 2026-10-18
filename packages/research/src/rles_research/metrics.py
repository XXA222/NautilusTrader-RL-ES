"""Economic-only metrics rebuilt from native account/fill facts; never from shaped reward."""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence, Mapping
from .numeric import finite_real


class MetricsError(ValueError):
    pass


def _number(v, name: str) -> float:
    if not finite_real(v):
        raise MetricsError(f'{name} must be finite numeric')
    try:
        result = float(v)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MetricsError(f'{name} must be finite numeric') from exc
    if not math.isfinite(result):
        raise MetricsError(f'{name} must be finite numeric')
    return result


def series_metrics(returns: Sequence[float], *, periods_per_year: float,
                   risk_free_per_period: float = 0.0, target_per_period: float = 0.0,
                   tail_quantile: float = 0.05) -> dict[str, object]:
    if (len(returns) < 2 or any(not finite_real(v)
                                for v in (periods_per_year, risk_free_per_period,
                                          target_per_period, tail_quantile))
            or periods_per_year <= 0 or not 0 < tail_quantile < 0.5):
        raise MetricsError('at least 2 returns and valid frequency/quantile required')
    r = [_number(x, 'return') for x in returns]
    if any(x <= -1 for x in r):
        raise MetricsError('returns <= -100% cannot be compounded safely')
    path = 1.0
    peak = 1.0
    dd = 0.0
    for x in r:
        path *= 1 + x
        if not math.isfinite(path) or path <= 0:
            raise MetricsError('equity compounding overflow/underflow')
        peak = max(peak, path)
        dd = max(dd, 1 - path / peak)
    downside = [min(0, x - target_per_period) ** 2 for x in r]
    downside_deviation = math.sqrt(statistics.fmean(downside))
    sortino = ((statistics.fmean(r) - target_per_period) / downside_deviation
               * math.sqrt(periods_per_year) if downside_deviation > 0 else None)
    sd = statistics.stdev(r)
    sharpe = ((statistics.fmean(r) - risk_free_per_period) / sd
              * math.sqrt(periods_per_year) if sd > 0 else None)
    count = max(1, math.ceil(len(r) * tail_quantile))
    worst = sorted(r)[:count]
    return {
        'net_return': path - 1,
        'max_drawdown': dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'mean_period_return': statistics.fmean(r),
        'std_period_return': sd,
        'undefined_metrics': [metric for metric, val in (('sharpe', sharpe), ('sortino', sortino))
                              if val is None],
        'value_at_risk': -worst[-1],
        'conditional_value_at_risk': -statistics.fmean(worst),
        'observations': len(r),
    }


class AccountFactsAccumulator:
    """Validate account rows incrementally without retaining normalized row dictionaries."""

    _REQUIRED = {'ts_ns', 'equity', 'external_cashflow', 'turnover_notional',
                 'fee', 'funding', 'slippage_cost'}

    def __init__(self, *, sample_interval_ns: int,
                 allow_price_improvement: bool = False) -> None:
        if type(allow_price_improvement) is not bool:
            raise MetricsError('allow_price_improvement must be Boolean')
        if (isinstance(sample_interval_ns, bool) or not isinstance(sample_interval_ns, int)
                or sample_interval_ns < 1):
            raise MetricsError('regular account snapshots and explicit sampling interval required')
        self.sample_interval_ns = sample_interval_ns
        self.allow_price_improvement = allow_price_improvement
        self.last_ts = -1
        self.previous: dict[str, float] | None = None
        self.first_equity = 0.0
        self.last_equity = 0.0
        self.external_cashflow = 0.0
        self.turnover = 0.0
        self.totals = {'fee': 0.0, 'funding': 0.0, 'slippage_cost': 0.0}
        # Exact VaR/CVaR order statistics require retaining returns. Other row
        # fields are reduced as they arrive and are not copied into a second list.
        self.returns: list[float] = []
        self.count = 0

    def add(self, fact: Mapping[str, object]) -> None:
        if not self._REQUIRED.issubset(fact):
            raise MetricsError('account facts missing required fields')
        ts = fact['ts_ns']
        if isinstance(ts, bool) or not isinstance(ts, int) or ts <= self.last_ts:
            raise MetricsError('account time order invalid')
        if self.last_ts != -1 and ts - self.last_ts != self.sample_interval_ns:
            raise MetricsError('irregular account sampling; annualization would be invalid')
        row = {key: _number(fact[key], key) for key in self._REQUIRED if key != 'ts_ns'}
        if (row['equity'] <= 0 or row['turnover_notional'] < 0 or row['fee'] < 0
                or (row['slippage_cost'] < 0 and not self.allow_price_improvement)):
            raise MetricsError('invalid account values / exhausted equity')
        if self.previous is None:
            self.first_equity = row['equity']
        else:
            adjusted = (row['equity'] - row['external_cashflow']) / self.previous['equity'] - 1
            if not math.isfinite(adjusted):
                raise MetricsError('cashflow-adjusted period return nonfinite')
            self.returns.append(adjusted)
            self.turnover += row['turnover_notional'] / self.previous['equity']
            for field in self.totals:
                self.totals[field] += row[field]
            self.external_cashflow += row['external_cashflow']
        self.previous = row
        self.last_equity = row['equity']
        self.last_ts = ts
        self.count += 1

    def result(self, *, periods_per_year: float,
               realized_trade_pnls: Sequence[float] | None = None) -> dict:
        if self.count < 3:
            raise MetricsError('regular account snapshots and explicit sampling interval required')
        result: dict[str, object] = series_metrics(self.returns, periods_per_year=periods_per_year)
        result['turnover_ratio'] = self.turnover
        for field, value in self.totals.items():
            result[f'total_{field}'] = value
        result['gross_return_before_reported_costs'] = (
            self.last_equity - self.first_equity - self.external_cashflow
            + self.totals['fee'] + self.totals['slippage_cost'] - self.totals['funding']
        ) / self.first_equity
        if realized_trade_pnls is not None:
            pnl = [_number(x, 'realized_pnl') for x in realized_trade_pnls]
            gross_profit = sum(x for x in pnl if x > 0)
            gross_loss = -sum(x for x in pnl if x < 0)
            result['trade_profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else None
            result['closed_trade_count'] = len(pnl)
        return result


def from_account_facts(facts: Iterable[Mapping[str, object]], *,
                       periods_per_year: float,
                       realized_trade_pnls: Sequence[float] | None = None,
                       sample_interval_ns: int,
                       allow_price_improvement: bool = False) -> dict:
    """Build economic metrics from regular native-account snapshots.

    Funding keeps its signed cashflow convention. External cashflow is treated as
    occurring at the sample boundary, so callers must split mid-period flows.
    All costs already entered native equity and must not be subtracted again.
    """
    accumulator = AccountFactsAccumulator(sample_interval_ns=sample_interval_ns,
                                          allow_price_improvement=allow_price_improvement)
    for fact in facts:
        accumulator.add(fact)
    return accumulator.result(periods_per_year=periods_per_year,
                              realized_trade_pnls=realized_trade_pnls)
