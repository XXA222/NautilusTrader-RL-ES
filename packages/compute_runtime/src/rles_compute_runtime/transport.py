"""Validate compact keyed messages without introducing cross-line Compute DTOs."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .errors import ComputeProtocolError

REQUIRED = ('run_id', 'attempt_id', 'env_id', 'episode_id', 'step_id', 'candidate_id', 'model_id', 'schema_version', 'dtype', 'shape')
MAX_BYTES = 1_048_576  # control-only; no large arrays/DataFrames/Engine states


def validate_key(key: Mapping[str, object]) -> dict[str, object]:
    if set(key) != set(REQUIRED):
        raise ComputeProtocolError(f'key must contain exactly {REQUIRED}')
    output: dict[str, object] = {}
    for field in REQUIRED:
        v = key[field]
        if field == 'shape':
            if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= 5 or any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in v):
                raise ComputeProtocolError('invalid tensor shape')
            output[field] = list(v)
            continue
        if field == 'dtype' and v not in {'float32','float64','int32','int64'}:
            raise ComputeProtocolError('unsupported dtype')
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            raise ComputeProtocolError(f'key field {field} must be string/int')
        if isinstance(v, str) and (not v or len(v) > 160):
            raise ComputeProtocolError(f'invalid key field {field}')
        if isinstance(v, int) and v < 0:
            raise ComputeProtocolError(f'negative key field {field}')
        output[field] = v
    return output


def small_json(value: object) -> object:
    """Refuse non-JSON data and oversized payloads before crossing a process boundary."""
    def reject(x):
        if isinstance(x, float) and not math.isfinite(x):
            raise ComputeProtocolError('NaN/Inf in worker payload')
        if isinstance(x, dict):
            if not all(isinstance(k, str) for k in x):
                raise ComputeProtocolError('keys must be strings')
            for v in x.values():
                reject(v)
        elif isinstance(x, list):
            for v in x:
                reject(v)
        elif x is not None and not isinstance(x, (str, int, float, bool)):
            raise ComputeProtocolError('worker payload must be JSON-only; pass dataset refs instead')
    reject(value)
    try:
        raw = json.dumps(value, allow_nan=False, separators=(',', ':')).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ComputeProtocolError('invalid JSON payload') from exc
    if len(raw) > MAX_BYTES:
        raise ComputeProtocolError('large payload rejected; use readonly data refs')
    return json.loads(raw)


def enforce_response(request_key: Mapping[str, object], response: Mapping[str, object]) -> object:
    if not isinstance(response, Mapping) or set(response) != {'key', 'status', 'payload'}:
        raise ComputeProtocolError('invalid response envelope')
    response_key = response['key']
    if not isinstance(response_key, Mapping):
        raise ComputeProtocolError('invalid response key')
    if validate_key(response_key) != validate_key(request_key):
        raise ComputeProtocolError('worker key mismatch; aborting attempt')
    if response['status'] != 'OK':
        raise ComputeProtocolError(f'worker signalled failure: {response["payload"]}')
    return small_json(response['payload'])
