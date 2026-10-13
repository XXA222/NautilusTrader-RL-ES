# NautilusTrader RL/ES Research

An experimental workspace for reinforcement learning and evolutionary strategies around NautilusTrader. The repository contains the RLES packages, tests, build metadata, and selected technical documentation.

This is research software. It does not promise profitable trading and is not financial advice. Backtest results depend on data quality, execution assumptions, fees, and risk settings.

## Scope

- Layered contracts, market-data validation, execution planning, risk controls, and NautilusTrader adapters.
- Gym environments, SB3-based RL, EvoTorch-based evolutionary strategies, hybrid workflows, and walk-forward research.
- Tests and small public-market-data fixtures used for regression coverage.

Large market datasets, credentials, trained models, local environments, and experiment outputs are intentionally excluded. Provide market data and private execution assumptions locally before running historical research.

## Setup

The workspace currently targets Python 3.14.7 and uses `uv`:

```powershell
uv sync --locked --all-packages --no-editable --extra application
uv run rles --help
uv run pytest tests -q
```

See `configs/ethusdt_20240101_real.json` for the configuration shape. Its referenced market data is not included. GPU tests require a compatible CUDA environment and are skipped when CUDA is unavailable.

## Repository access

The repository is public for browsing and cloning. Only the repository owner has push permission; outside users cannot push changes to this repository.
