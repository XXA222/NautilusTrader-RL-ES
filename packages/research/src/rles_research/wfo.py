"""WFO window requests, validated against L2-owned immutable FoldManifest fields."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Mapping
from typing import cast
import re


class FoldBoundaryError(ValueError):
    pass


def _utc(t: datetime) -> datetime:
    if (not isinstance(t, datetime) or t.tzinfo is None
            or t.utcoffset() != timedelta(0)):
        raise FoldBoundaryError('UTC-aware datetime required')
    return t


def _ns(t: datetime) -> int:
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    delta=t-epoch
    return ((delta.days*86400+delta.seconds)*1_000_000_000 + delta.microseconds*1000)


def _duration_ns(value: timedelta) -> int:
    return ((value.days * 86400 + value.seconds) * 1_000_000_000
            + value.microseconds * 1000)


@dataclass(frozen=True)
class FoldRequest:
    fold_id: str
    train_start_ns: int
    train_end_ns: int
    validation_start_ns: int
    validation_end_ns: int
    test_start_ns: int
    test_end_ns: int
    warmup_start_ns: int
    purge_ns: int
    embargo_ns: int
    terminal_policy: str = 'force_flatten_with_cost'

    def __post_init__(self) -> None:
        if (not isinstance(self.fold_id, str)
                or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', self.fold_id) is None):
            raise FoldBoundaryError('invalid fold identity')
        clocks = (self.warmup_start_ns, self.train_start_ns, self.train_end_ns,
                  self.validation_start_ns, self.validation_end_ns,
                  self.test_start_ns, self.test_end_ns)
        if any(type(value) is not int or not 1 <= value <= 2**63 - 1 for value in clocks):
            raise FoldBoundaryError('fold boundaries must be positive integer UTC nanoseconds')
        if not (self.warmup_start_ns < self.train_start_ns < self.train_end_ns <=
                self.validation_start_ns < self.validation_end_ns <= self.test_start_ns <
                self.test_end_ns):
            raise FoldBoundaryError('fold windows overlap or are not strictly ordered')
        if (type(self.purge_ns) is not int or type(self.embargo_ns) is not int
                or not 0 <= self.purge_ns <= 2**63 - 1
                or not 0 <= self.embargo_ns <= 2**63 - 1):
            raise FoldBoundaryError('purge/embargo must be nonnegative integer nanoseconds')
        gap = self.purge_ns + self.embargo_ns
        if (self.train_end_ns + gap > self.validation_start_ns
                or self.validation_end_ns + gap > self.test_start_ns):
            raise FoldBoundaryError('fold score windows violate purge/embargo separation')
        if self.terminal_policy != 'force_flatten_with_cost':
            raise FoldBoundaryError('scored folds require force_flatten_with_cost')


def plan_folds(*, start: datetime, end: datetime, scheme: str, warmup: timedelta,
               purge: timedelta, embargo: timedelta, step_days: int=15,
               final_holdout_start: datetime|None=None,
               data_available_from: datetime|None=None) -> list[FoldRequest]:
    """Train score excludes warmup; gap between train/val/test includes purge+embargo.

    The L2 builder supplies the final immutable FoldManifest; this is only a request.
    """
    if scheme not in {'60/15/15','360/15/15'} or type(step_days) is not int or step_days != 15:
        raise FoldBoundaryError('only frozen 60/15/15 and 360/15/15, step=15, supported')
    if (any(type(x) is not timedelta for x in (warmup, purge, embargo))
            or any(x < timedelta(0) for x in (warmup,purge,embargo))
            or warmup <= timedelta(0)):
        raise FoldBoundaryError('explicit positive warmup and nonnegative purge/embargo required')
    _utc(start);_utc(end)
    if end <= start:
        raise FoldBoundaryError('dataset end must follow start')
    if data_available_from is None or _utc(data_available_from) > start-warmup:
        raise FoldBoundaryError('L2 data availability must cover the entire warmup context')
    horizon=min(end,_utc(final_holdout_start)) if final_holdout_start else end
    if horizon<=start:raise FoldBoundaryError('invalid dataset/holdout boundary')
    train_days=int(scheme.split('/')[0]); gap=purge+embargo
    window=start
    folds: list[FoldRequest] = []
    while True:
        tr_end=window+timedelta(days=train_days)
        v_start=tr_end+gap
        v_end=v_start+timedelta(days=15)
        t_start=v_end+gap
        t_end=t_start+timedelta(days=15)
        if t_end>horizon:break
        request=FoldRequest(f'{scheme.replace("/","-")}-{len(folds):04d}',
             _ns(window),_ns(tr_end),_ns(v_start),_ns(v_end),_ns(t_start),_ns(t_end),
             _ns(window-warmup),_duration_ns(purge),_duration_ns(embargo))
        folds.append(request)
        window+=timedelta(days=step_days)
    if not folds:raise FoldBoundaryError('dataset too short for one leakage-safe fold')
    return folds


def validate_l2_manifest(request: FoldRequest, manifest: object, *,
                         expected_dataset_hash: str | None = None,
                         expected_feature_hash: str | None = None,
                         expected_scaler_hash: str | None = None) -> None:
    """Validate L2's actual nested FoldManifest without importing ``rles_data``.

    The consumer accepts the frozen L2 dataclass, its dataclass/asdict mapping,
    or the earlier flat carrier. This is structural validation, not a substitute
    for binding the manifest hash into the run fingerprint.
    """
    missing = object()

    def field(value: object, name: str) -> object:
        if isinstance(value, Mapping):
            return value.get(name, missing)
        return getattr(value, name, missing)

    flat_fields = tuple(request.__dataclass_fields__)
    if all(field(manifest, name) is not missing for name in flat_fields):
        actual = {name: field(manifest, name) for name in flat_fields}
    else:
        windows = {name: field(manifest, name) for name in ('train', 'validation', 'test')}
        if any(window is missing for window in windows.values()):
            raise FoldBoundaryError('L2 FoldManifest must provide nested train/validation/test windows')
        try:
            actual = {
                'warmup_start_ns': field(windows['train'], 'warmup_start_ns'),
                'train_start_ns': field(windows['train'], 'score_start_ns'),
                'train_end_ns': field(windows['train'], 'score_end_ns'),
                'validation_start_ns': field(windows['validation'], 'score_start_ns'),
                'validation_end_ns': field(windows['validation'], 'score_end_ns'),
                'test_start_ns': field(windows['test'], 'score_start_ns'),
                'test_end_ns': field(windows['test'], 'score_end_ns'),
                'purge_ns': field(manifest, 'purge_ns'),
                'embargo_ns': field(manifest, 'embargo_ns'),
                'terminal_policy': field(manifest, 'terminal_policy'),
            }
        except (AttributeError, TypeError) as exc:
            raise FoldBoundaryError('malformed nested L2 FoldManifest') from exc
        if any(value is missing for value in actual.values()):
            raise FoldBoundaryError('nested L2 FoldManifest is missing required boundary fields')
        if field(manifest, 'schema_version') != '0.4.0':
            raise FoldBoundaryError('unsupported L2 FoldManifest schema version')
        warmup_ns = request.train_start_ns - request.warmup_start_ns
        expected_windows = {
            'train': (request.warmup_start_ns, request.train_start_ns, request.train_end_ns),
            'validation': (request.validation_start_ns - warmup_ns,
                           request.validation_start_ns, request.validation_end_ns),
            'test': (request.test_start_ns - warmup_ns, request.test_start_ns,
                     request.test_end_ns),
        }
        for name, window in windows.items():
            window_values = tuple(field(window, key) for key in
                                  ('warmup_start_ns', 'score_start_ns', 'score_end_ns'))
            if any(type(value) is not int or not 1 <= cast(int, value) <= 2**63 - 1
                   for value in window_values):
                raise FoldBoundaryError(f'malformed L2 {name} window')
            warmup_start, score_start, score_end = cast(tuple[int, int, int], window_values)
            if not warmup_start <= score_start < score_end:
                raise FoldBoundaryError(f'malformed L2 {name} window')
            if window_values != expected_windows[name]:
                boundary_name = {
                    'train': ('warmup_start_ns', 'train_start_ns', 'train_end_ns'),
                    'validation': ('validation_start_ns', 'validation_start_ns', 'validation_end_ns'),
                    'test': ('test_start_ns', 'test_start_ns', 'test_end_ns'),
                }[name]
                for index, key in enumerate(boundary_name):
                    if window_values[index] != expected_windows[name][index]:
                        raise FoldBoundaryError(f'L2 FoldManifest drift at {key}; submit ICR')
            if warmup_start != score_start - warmup_ns:
                raise FoldBoundaryError(f'L2 {name} warmup duration drift')
        for name in ('dataset_hash', 'feature_hash'):
            value = field(manifest, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(ch not in '0123456789abcdef' for ch in value)):
                raise FoldBoundaryError(f'L2 FoldManifest has invalid {name}')
        scaler_hash = field(manifest, 'scaler_hash')
        if (scaler_hash is not None and (not isinstance(scaler_hash, str)
                or len(scaler_hash) != 64
                or any(ch not in '0123456789abcdef' for ch in scaler_hash))):
            raise FoldBoundaryError('L2 FoldManifest has invalid scaler_hash')
        if any(field(manifest, key) is not True for key in
               ('reset_positions', 'reset_orders', 'reset_replay', 'freeze_validation_scaler')):
            raise FoldBoundaryError('L2 fold state/scaler isolation flags must all be true')

    # L2 FoldManifest has no fold_id field; the stable request ID is assigned
    # by this consumer from its deterministic window order.
    for key in flat_fields:
        if key not in actual:
            continue
        value, expected = actual[key], getattr(request, key)
        expected_type = int if key.endswith('_ns') else str
        if type(value) is not expected_type or value != expected:
            raise FoldBoundaryError(f'L2 FoldManifest drift at {key}; submit ICR')

    for name, expected in (('dataset_hash', expected_dataset_hash),
                           ('feature_hash', expected_feature_hash),
                           ('scaler_hash', expected_scaler_hash)):
        if expected is not None and field(manifest, name) != expected:
            raise FoldBoundaryError(f'L2 FoldManifest fingerprint drift at {name}')
