"""Economic-only metrics rebuilt from native account/fill facts; never from shaped reward."""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence,Mapping
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
                   risk_free_per_period: float=0.0, target_per_period: float=0.0,
                   tail_quantile: float=0.05) -> dict[str, object]:
    if (len(returns)<2 or any(not finite_real(v)
                                 for v in (periods_per_year,risk_free_per_period,target_per_period,tail_quantile))
        or periods_per_year<=0 or not 0<tail_quantile<0.5):
        raise MetricsError('at least 2 returns and valid frequency/quantile required')
    r=[_number(x,'return') for x in returns]
    if any(x<=-1 for x in r):
        raise MetricsError('returns <= -100% cannot be compounded safely')
    path=1.;peak=1.;dd=0.
    for x in r:
        path*=1+x
        if not math.isfinite(path) or path<=0:
            raise MetricsError('equity compounding overflow/underflow')
        peak=max(peak,path); dd=max(dd,1-path/peak)
    downside=[min(0,x-target_per_period)**2 for x in r]
    downside_deviation=math.sqrt(statistics.fmean(downside))
    sortino=((statistics.fmean(r)-target_per_period)/downside_deviation*math.sqrt(periods_per_year)
             if downside_deviation>0 else None)
    sd=statistics.stdev(r)
    sharpe=((statistics.fmean(r)-risk_free_per_period)/sd*math.sqrt(periods_per_year)
            if sd>0 else None)
    count=max(1,math.ceil(len(r)*tail_quantile))
    worst=sorted(r)[:count]
    result: dict[str, object] = {'net_return':path-1,'max_drawdown':dd,'sharpe':sharpe,'sortino':sortino,
            'mean_period_return':statistics.fmean(r),'std_period_return':sd,
            'undefined_metrics':[metric for metric,val in [('sharpe',sharpe),('sortino',sortino)] if val is None],
            'value_at_risk':-worst[-1],'conditional_value_at_risk':-statistics.fmean(worst),
            'observations':len(r)}
    return result


def from_account_facts(facts: Sequence[Mapping[str,object]], *,
                       periods_per_year:float, realized_trade_pnls:Sequence[float]|None=None,
                       sample_interval_ns:int, allow_price_improvement: bool = False) -> dict:
    """Build economic metrics from regular native-account snapshots.

    Each row contains ``ts_ns`` and end-of-period ``equity`` plus period-flow
    increments for ``external_cashflow``, ``turnover_notional``, ``fee``,
    ``funding`` and ``slippage_cost``. Funding keeps its signed native cashflow
    convention (receipts positive, payments negative); fees are nonnegative and
    signed slippage records either execution cost or price improvement. External cashflow is treated as occurring at the sample
    boundary, so callers must split intervals when a flow occurs mid-period.

    All costs already entered native equity; no second subtraction is allowed.
    """
    if type(allow_price_improvement) is not bool:
        raise MetricsError('allow_price_improvement must be Boolean')
    if len(facts)<3 or isinstance(sample_interval_ns,bool) or not isinstance(sample_interval_ns,int) or sample_interval_ns<1:
        raise MetricsError('regular account snapshots and explicit sampling interval required')
    rows=[];last_ts=-1
    required={'ts_ns','equity','external_cashflow','turnover_notional','fee','funding','slippage_cost'}
    for f in facts:
        if not required.issubset(f):raise MetricsError('account facts missing required fields')
        ts=f['ts_ns']
        if isinstance(ts,bool) or not isinstance(ts,int) or ts<=last_ts:
            raise MetricsError('account time order invalid')
        if last_ts != -1 and ts-last_ts != sample_interval_ns:
            raise MetricsError('irregular account sampling; annualization would be invalid')
        last_ts=ts
        row={k:_number(f[k],k) for k in required if k!='ts_ns'}
        if (row['equity']<=0 or row['turnover_notional']<0 or row['fee']<0
                or (row['slippage_cost']<0 and not allow_price_improvement)):
            raise MetricsError('invalid account values / exhausted equity')
        rows.append(row)
    returns=[];turnover=0.0
    for previous,current in zip(rows,rows[1:]):
        adjusted=(current['equity']-current['external_cashflow'])/previous['equity']-1
        if not math.isfinite(adjusted):
            raise MetricsError('cashflow-adjusted period return nonfinite')
        returns.append(adjusted)
        turnover+=current['turnover_notional']/previous['equity']
    result: dict[str, object] = series_metrics(returns,periods_per_year=periods_per_year)
    result['turnover_ratio']=turnover
    totals: dict[str, float] = {}
    for field in ('fee','funding','slippage_cost'):
        totals[field] = sum(r[field] for r in rows[1:])
        result[f'total_{field}'] = totals[field]
    result['gross_return_before_reported_costs']=(rows[-1]['equity']-rows[0]['equity']-
        sum(row['external_cashflow'] for row in rows[1:])+
        totals['fee']+totals['slippage_cost']-totals['funding'])/rows[0]['equity']
    if realized_trade_pnls is not None:
        pnl=[_number(x,'realized_pnl') for x in realized_trade_pnls]
        gross_profit=sum(x for x in pnl if x>0)
        gross_loss=-sum(x for x in pnl if x<0)
        result['trade_profit_factor']=gross_profit/gross_loss if gross_loss>0 else None
        result['closed_trade_count']=len(pnl)
    return result
