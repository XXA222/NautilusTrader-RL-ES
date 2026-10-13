"""Constraint-first immutable model selection on validation metrics only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping,Sequence
from .numeric import finite_real


def _as_metric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('candidate metric is not numeric')
    return float(value)


@dataclass(frozen=True)
class SelectionSpec:
    max_drawdown: float
    max_turnover: float
    max_risk_violations: int
    primary_metric: str
    maximize: bool = True
    min_median_fold_net_return: float = 0.0

    def __post_init__(self):
        if (any(not finite_real(x)
                for x in (self.max_drawdown,self.max_turnover,self.max_risk_violations,
                          self.min_median_fold_net_return))
            or type(self.max_risk_violations) is not int or type(self.maximize) is not bool
            or not (0<self.max_drawdown<=1 and self.max_turnover>0 and self.max_risk_violations>=0)):
            raise ValueError('invalid constraints')
        if self.primary_metric not in {'median_fold_sortino','median_fold_net_return','median_fold_sharpe'}:
            raise ValueError('metric must be frozen before Test/Holdout')

    def sha256(self):
        payload={field:getattr(self,field) for field in self.__dataclass_fields__}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def choose(spec: SelectionSpec, candidates: Sequence[Mapping[str,object]], *,
           fold_stage: str='validation') -> dict:
    if fold_stage!='validation':
        raise ValueError('selection on Test or Holdout is prohibited')
    if not candidates:raise ValueError('no validation candidates')
    accepted: list[dict[str, object]] = [];seen=set();rejected={}
    needed={'candidate_id','liquidations','max_drawdown','turnover','risk_violations',
            'median_fold_net_return',spec.primary_metric}
    for c in candidates:
        if not needed.issubset(c):raise ValueError('incomplete validation candidate evidence')
        cid=c['candidate_id']
        if not isinstance(cid,str) or not cid or cid in seen:raise ValueError('duplicate candidate')
        seen.add(cid)
        normalized: dict[str, object] = {'candidate_id': cid}
        for k in needed-{'candidate_id'}:
            value = c[k]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('nonfinite candidate metric')
            if not finite_real(value):
                raise ValueError('nonfinite candidate metric')
            numeric_value = float(value)
            normalized[k] = numeric_value
        metrics = {key: _as_metric(normalized[key]) for key in needed-{'candidate_id'}}
        if (any(metrics[name]<0 or not metrics[name].is_integer()
                for name in ('liquidations','risk_violations'))
            or metrics['max_drawdown']<0 or metrics['turnover']<0):
            raise ValueError('invalid nonnegative constraint metric')
        if metrics['median_fold_net_return'] <= 0:
            rejected[cid] = 'NONPOSITIVE_NET_RETURN'
        elif (metrics['liquidations']!=0 or metrics['max_drawdown']>spec.max_drawdown
            or metrics['turnover']>spec.max_turnover
            or metrics['risk_violations']>spec.max_risk_violations
            or metrics['median_fold_net_return']<spec.min_median_fold_net_return):
            rejected[cid]='CONSTRAINT_VIOLATION'
        else:accepted.append(normalized)
    if not accepted:return {'status':'NO_FEASIBLE_MODEL','rejected':rejected,'selection_spec_sha256':spec.sha256()}
    # tie-break uses stable candidate ID; never silently depend on query insertion order
    accepted.sort(key=lambda c:((-1 if spec.maximize else 1)*_as_metric(c[spec.primary_metric]),
                                str(c['candidate_id'])))
    return {'status':'SELECTED','candidate_id':accepted[0]['candidate_id'],
            'selected_primary_metric':_as_metric(accepted[0][spec.primary_metric]),
            'rejected':rejected,'selection_spec_sha256':spec.sha256()}
