"""Frozen *requests* for adversarial research, not a market-data simulator.

L2/L3 must execute these scenarios through their authoritative data / engine /
execution / hard-risk providers. This module never mutates OHLCV or settlement.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


class StressProtocolError(ValueError):
    pass


# Named by v0.4.0 §14.2–14.3. Severity/calibration of shocks is external and
# must be fixed in the original L2/L3 risk/execution experiment manifests.
_SCENARIOS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ('latency','latency_x2',{'latency_multiplier':2}),
    ('latency','latency_x5',{'latency_multiplier':5}),
    ('slippage','conservative',{}),
    ('slippage','extreme',{}),
    ('cost','fee_increase',{}),
    ('cost','funding_extreme',{}),
    ('cost','funding_sign_flip',{}),
    ('data','missing_1m',{}),
    ('data','delayed_mark',{}),
    ('data','missing_funding',{}),
    ('data','instrument_spec_change',{}),
    ('price','gap',{}),
    ('price','rapid_reversal',{}),
    ('price','mark_last_divergence',{}),
    ('exploit','zero_volatility_turnover_cost',{}),
    ('exploit','future_extrema_perturbation',{}),
    ('exploit','future_funding_swap',{}),
    ('exploit','terminal_boundary_shift',{}),
    ('exploit','legal_order_permutation',{}),
    ('exploit','action_sign_scale',{}),
)
_HASH_NAMES=('dataset_sha256','execution_sha256','risk_sha256','feature_sha256',
             'actor_signature_sha256','seed_tree_sha256')


def _scenario_parameters(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StressProtocolError('scenario parameters must be a string-keyed mapping')
    return dict(value)


def _hash(obj:object) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class StressPlan:
    protocol_id: str
    frozen_fingerprints: Mapping[str,str]
    scenarios: tuple[Mapping[str,object], ...]
    plan_sha256: str

    def as_mapping(self) -> dict[str,object]:
        return {'schema_version':1,'protocol_id':self.protocol_id,
                'frozen_fingerprints':dict(self.frozen_fingerprints),
                'scenarios':[{**dict(x),'parameters':_scenario_parameters(x['parameters'])}
                             for x in self.scenarios],
                'plan_sha256':self.plan_sha256,
                'external_execution_required':True}


def plan_stress_protocol(protocol_id: str, frozen_fingerprints: Mapping[str,str]) -> StressPlan:
    """Build a deterministic immutable scenario catalog for L2/L3 providers.

    Scenario parameters not explicitly fixed in v0.4.0 are deliberately absent
    instead of being guessed. Integration must use frozen shock config hashes.
    """
    if not isinstance(protocol_id,str) or not 0<len(protocol_id)<=128:
        raise StressProtocolError('nonempty, bounded protocol ID required')
    if (not isinstance(frozen_fingerprints,Mapping) or set(frozen_fingerprints)!=set(_HASH_NAMES)
        or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
               for v in frozen_fingerprints.values())):
        raise StressProtocolError('all six immutable dataset/actor/risk/execution fingerprints required')
    specs: tuple[dict[str, object], ...] = tuple({'scenario_id':f'{category}.{name}','category':category,'name':name,
                 'parameters':dict(parameters),
                 'hard_risk_frozen':True, 'terminal_policy':'force_flatten_with_cost',
                 'available_at_enforced':True}
                for category,name,parameters in _SCENARIOS)
    payload={'schema_version':1,'protocol_id':protocol_id,
             'frozen_fingerprints':dict(frozen_fingerprints),
             'scenarios':list(specs)}
    readonly_specs = tuple(
        MappingProxyType({**x, 'parameters': MappingProxyType(_scenario_parameters(x['parameters']))})
        for x in specs
    )
    return StressPlan(protocol_id,MappingProxyType(dict(frozen_fingerprints)),readonly_specs,_hash(payload))
