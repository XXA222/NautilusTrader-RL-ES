"""Deterministic B0 next-return distributions and CRPS scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Sequence

from rles_data.core import DataError, FoldManifest, MINUTE_NS

from .contracts import MarketReplay


@dataclass(frozen=True)
class BaselineScore:
    name: str
    observations: int
    mean_crps: float
    horizon_minutes: int = 1


def _log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(right / left) for left, right in zip(closes, closes[1:])]


def crps_ensemble(samples: Sequence[float], observed: float) -> float:
    """Exact empirical CRPS: E|X-y| - 1/2 E|X-X'|."""
    if not samples or not math.isfinite(observed) or any(not math.isfinite(x) for x in samples):
        raise DataError("CRPS requires finite observations and a nonempty sample ensemble")
    ordered = sorted(samples)
    n = len(ordered)
    first = sum(abs(value - observed) for value in ordered) / n
    pair_sum = sum((2 * index - n + 1) * value for index, value in enumerate(ordered))
    return first - pair_sum / (n * n)


def stationary_block_bootstrap(values: Sequence[float], *, paths: int, horizon: int,
                               mean_block_length: int, rng: random.Random
                               ) -> tuple[tuple[float, ...], ...]:
    """Draw circular stationary-bootstrap paths with geometric block lengths."""
    if (not values or paths < 1 or horizon < 1 or mean_block_length < 1
            or any(not math.isfinite(value) for value in values)):
        raise DataError("invalid stationary block-bootstrap inputs")
    restart_probability = 1.0 / mean_block_length
    result = []
    for _ in range(paths):
        index = rng.randrange(len(values))
        path = []
        for _ in range(horizon):
            path.append(values[index])
            if rng.random() < restart_probability:
                index = rng.randrange(len(values))
            else:
                index = (index + 1) % len(values)
        result.append(tuple(path))
    return tuple(result)


def score_baselines(replay: MarketReplay, fold: FoldManifest, *, seed: int = 0,
                    ensemble_size: int = 128, volatility_lookback: int = 60,
                    bootstrap_block: int = 16,
                    ewma_decay: float = 0.94) -> tuple[BaselineScore, ...]:
    """Score S0/S1/S2 over Validation using Train-only context/statistics.

    Validation closes are read only as scoring targets. All fitted quantities and
    S2 source residuals come from the Fold's Train score window.
    """
    if (ensemble_size < 2 or volatility_lookback < 2 or bootstrap_block < 1
            or not 0 < ewma_decay < 1):
        raise DataError("invalid baseline sampling configuration")
    if not replay.is_contiguous:
        raise DataError("S0/S1/S2 scoring requires a gap-free replay")
    instrument = replay.frames[0].instrument
    if any(frame.instrument != instrument for frame in replay.frames):
        raise DataError("baseline replay instrument mismatch")
    train = [frame for frame in replay.frames
             if fold.train.score_start_ns <= frame.ts_close_ns < fold.train.score_end_ns]
    validation_context = [frame for frame in replay.frames
                         if fold.validation.warmup_start_ns <= frame.ts_close_ns
                         < fold.validation.score_start_ns]
    validation = [frame for frame in replay.frames
                  if fold.validation.score_start_ns <= frame.ts_close_ns < fold.validation.score_end_ns]
    if (len(train) < max(volatility_lookback + 1, bootstrap_block + 1)
            or len(validation_context) < volatility_lookback + 1 or not validation):
        raise DataError("fold lacks enough Train/Validation bars for S0/S1/S2")
    if any(frame.available_at_ns > frame.ts_close_ns + MINUTE_NS
           for frame in train + validation_context + validation):
        raise DataError("bar availability exceeds the configured one-minute audit allowance")
    if (any(right.ts_close_ns - left.ts_close_ns != MINUTE_NS
            for left, right in zip(validation_context, validation_context[1:]))
            or validation_context[-1].ts_close_ns + MINUTE_NS != validation[0].ts_close_ns
            or any(right.ts_close_ns - left.ts_close_ns != MINUTE_NS
                   for left, right in zip(validation, validation[1:]))):
        raise DataError("Validation warmup and score windows must be contiguous 1m candles")
    if validation[0].ts_close_ns <= validation_context[-1].available_at_ns:
        raise DataError("validation target begins before its causal warmup is available")

    train_returns = _log_returns([frame.close for frame in train])
    context_returns = _log_returns([frame.close for frame in validation_context])
    validation_returns = [math.log(validation[0].close / validation_context[-1].close)]
    validation_returns.extend(_log_returns([frame.close for frame in validation]))
    rng = random.Random(seed)
    scores: dict[str, list[float]] = {"S0_ZERO_DRIFT": [], "S1_CONDITIONAL_VOL": [],
                                     "S2_BLOCK_BOOTSTRAP": []}
    # At validation start, only prior observed Train returns can seed the models.
    train_mean = fmean(train_returns)
    train_scale = max(math.sqrt(fmean(value * value for value in train_returns)), 1e-12)
    train_innovations = tuple((value - train_mean) / train_scale for value in train_returns)
    ewma_variance = train_scale * train_scale
    for value in train_returns:
        ewma_variance = ewma_decay * ewma_variance + (1.0 - ewma_decay) * value * value
    for value in context_returns[-volatility_lookback:]:
        ewma_variance = ewma_decay * ewma_variance + (1.0 - ewma_decay) * value * value
    for observed in validation_returns:
        samples_zero = [rng.gauss(0.0, train_scale) for _ in range(ensemble_size)]
        conditional_scale = max(math.sqrt(ewma_variance), 1e-12)
        samples_vol = [rng.choice(train_innovations) * conditional_scale
                       for _ in range(ensemble_size)]
        bootstrap_paths = stationary_block_bootstrap(
            train_returns, paths=ensemble_size, horizon=1,
            mean_block_length=bootstrap_block, rng=rng,
        )
        samples_boot = [path[0] for path in bootstrap_paths]
        scores["S0_ZERO_DRIFT"].append(crps_ensemble(samples_zero, observed))
        scores["S1_CONDITIONAL_VOL"].append(crps_ensemble(samples_vol, observed))
        scores["S2_BLOCK_BOOTSTRAP"].append(crps_ensemble(samples_boot, observed))
        # Online conditional prediction after the target is revealed is causal.
        ewma_variance = ewma_decay * ewma_variance + (1.0 - ewma_decay) * observed * observed
    return tuple(BaselineScore(name, len(values), fmean(values))
                 for name, values in scores.items())
