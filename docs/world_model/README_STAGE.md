# Route B World Model — B0/B1

This folder records the first Route B engineering increment. It does not claim
that the market model is trained, calibrated, economically useful, or ready for
trading.

## Implemented

- `packages/world_model` defines immutable 1m market frames and replay contracts,
  projects canonical `rles_data` events, and filters visible bars at an explicit
  availability barrier.
- B0 audits timestamp gaps and duplicates. S0 uses zero drift with Train-only
  fixed volatility; S1 uses EWMA volatility with Train-only standardized
  innovations; S2 samples stationary-bootstrap paths with geometric block
  lengths and scores their one-step marginal. All three use empirical one-step
  log-return CRPS.
- Validation scoring uses the Fold's causal warmup and score windows. Scaling
  statistics and bootstrap source returns come only from Train. The current B0
  baseline is one-step only; it does not establish multi-step path fidelity.
- `rles world-model ...` is routed through the existing CLI. `audit-data
  --registered-train-config` calls `project_registered_train_replay`, which
  reserves the project Train scope before reading a curated payload. No
  production parquet was read for this increment. Unregistered JSON replay is
  accepted only when explicitly marked `synthetic_test_only`.
- B1 keeps the JAX learner in a separate Python 3.12.10 environment under
  `packages/world_model/environment/`. The main Python 3.14.7 environment has no
  JAX dependency. DreamerV3 source is checked out in ignored `.tmp` at the exact
  design commit; its patch manifest is empty. The adapter calls upstream RSSM
  `observe` and `imagine` with an all-zero `market_dummy` action and never
  forwards policy action values.

## Verify

From the repository root (the source package paths are included by the project
environment when installed):

```powershell
uv sync --project packages/orchestrator --extra application
uv run --project packages/orchestrator --extra application rles world-model audit-data --synthetic-replay-json <synthetic-replay.json>
uv run --project packages/orchestrator --extra application rles world-model audit-data --registered-train-config <registered-train-config.json>
uv run --project packages/orchestrator --extra application pytest -q tests/world_model
.tmp/world-model-cpu-venv/Scripts/python.exe qa/world_model/b1_cpu_smoke.py --checkout .tmp/dreamerv3-route-b --output .tmp/world-model-b1-cpu-smoke.json
```

The `synthetic_test_only` replay CLI does not confer research-data authorization.
Real data must be read through the registered Train reader. A Validation
baseline run still requires a separately authorized, frozen evaluation-scope
reader; until that interface is implemented, production baseline scoring stays
blocked. Do not use Test/Holdout or previously consumed windows to create a new
Route B result.

## Current gate

- **B0 engineering:** PASS for contract, causal-barrier, gap-rejection, CRPS, and
  S0/S1/S2 mechanics in synthetic unit tests.
- **B0 data/baseline evidence:** BLOCKED. No fresh Route B Train/Validation
  allocation and frozen dataset/fold identity were used, so the reports in this
  folder contain no real market audit or baseline score.
- **B1 CPU:** PASS on Windows Python 3.12.10 with JAX 0.11.2, pinned upstream
  DreamerV3, same-seed action-independence, and NPZ state roundtrip.
- **WSL2/CUDA:** BLOCKED by the host returning `Wsl/Service/E_ACCESSDENIED` before
  a WSL command could run. This is not a GPU failure result; no WSL device probe
  occurred.
- **B2:** Not started. B0's real-data gate has not passed. No Actor, LIMIT path,
  policy imagination, or economic claim is implemented.

Evidence: [`B0_data_audit.json`](B0_data_audit.json),
[`baselines.json`](baselines.json), [`B1_env_lock.json`](B1_env_lock.json), and
[`B1_cpu_smoke.json`](B1_cpu_smoke.json).
