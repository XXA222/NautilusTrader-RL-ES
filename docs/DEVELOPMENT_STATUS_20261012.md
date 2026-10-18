# Nautilus-RLES development checkpoint — 2026-10-12

## Implemented in the current source

- L2 interval coverage now verifies mark/index/funding freshness, every expected
  funding settlement, and continuous historical instrument-rule validity. Full
  RL/ES capability cannot be granted from event presence alone.
- Observation schema inputs are copied to immutable tuples. R1 transfer and
  distillation explicitly convert raw observations through the frozen L4
  normalization transform.
- The Binance public archive importer verifies source checksums and imports
  closed ETHUSDT one-minute bars, mark/index closes and settled funding with a
  declared availability delay. Its new inclusive date-range path fetches daily
  market streams and monthly funding streams once each, without inventing rules.
- The historical L3 factory builds fresh native NautilusTrader perpetual
  sessions, conservative bar slippage, latency, maker/taker fees, margin,
  terminal flattening, 23-column observations including signed and gross
  pending-order exposure, and frozen Train-inner
  feature scaling. Economic event time assigns fold membership; availability
  time remains the decision clock.
- Native execution supports explicit `bar` and `quote` modes. Quote mode uses
  Nautilus matching with displayed quote liquidity consumption, records the
  as-of opposing best quote at submission, rejects quotes older than two
  seconds, and requires zero synthetic BPS slippage. A synthetic engine test
  verifies that a 0.5 ETH order consumes only 0.1 ETH at the first displayed
  ask and fills its remainder from a later quote.
- Official one-day 2024-01-01 Binance bookTicker data was replayed through the
  native engine without orders: 90,718 native market events in 87,835 timestamp
  batches completed in about 0.85 seconds. This proves feed/clock integration
  only; the harness used test rules, emitted no trades, and is not a real
  economic backtest.
- Settled funding facts keep their source settlement timestamp. If the archived
  row becomes available later, the native next-settlement boundary is scheduled
  at `available_at + 1 ns`, avoiding use of a published settled rate before it
  was available. Unit coverage verifies both clocks. This derived scheduling
  policy still needs comparison with an independent account-ledger golden.
- The installed `rles` application supports market audit, validation, training
  and backtest for PPO/SAC/TD3/DDPG and A0/A1/A4 routes. A1 and A4 use the
  installed spawn-worker environment factory. Training evaluation is restricted
  to Validation; backtest accepts Test or Holdout.
- Native account snapshots are converted to regular economic rows with
  event-identity deduplication for fills and funding. Fees and signed funding are
  taken from native facts, and reported costs are not subtracted from equity a
  second time. The perpetual factory now enables Nautilus liquidation simulation
  and counts orders from native venue tags/IDs. A streaming engine trigger test
  observed the generated order and fill; Binance-specific equivalence is not certified.
- Trading bundles bind dataset, feature, schema, scaler, execution, hard-risk
  and application identities. Bundle/report files carry SHA-256 sidecars; actor
  loading, bundle integrity and actual replay identities are verified before a
  backtest.
- Unsupported CUDA ES checkpoint execution is rejected before candidate
  evaluation or budget consumption.
- Before the bookTicker changes, an 11-wheel set was rebuilt from the then-current source under
  `.tmp/wheel-release-observation-f05`; `SHA256SUMS.json` records every wheel's
  hash and size. All 11 wheels were installed into the isolated target
  `.tmp/wheel-release-installed-observation-f05`, and key imports resolved to
  that target. The canonical `dist-current` contains two ACL-locked wheels that
  could not be read or replaced reliably. A clean, separately named current
  set is available under `dist-current-verified-20261012`; its 11 wheel hashes
  were rechecked against the build manifest. Keep F13 open until the canonical
  directory can be safely synchronized and validated.

## Verification completed

- Full CPU suite after the pending-order observation extension (before quote support): **240 passed,
  4 CUDA-only skipped**.
- Static checks: **MyPy 97 source files passed; Ruff passed**.
- `tests/test_historical_factory.py`: **10 passed**, including signed/gross
  pending-exposure values and rejection of invalid native order side/quantity.
- Historical application tests: four RL algorithms plus A1 and A4 train,
  export, reload and native backtest; bundle tamper rejection is exercised.
- The full CPU rerun covers native reporting and fail-closed unknown-count
  behavior. A streaming native engine regression triggers a liquidation order;
  the adapter records its native venue tag, order ID, and fill.
- Isolated wheel target: CPython 3.14.7, 11 current project wheels installed;
  key package imports resolved to the target, and the full CPU suite passed
  from the installed wheels (**240 passed, 4 CUDA-only skipped**). The target
  used the already-installed locked third-party dependencies; it was not a
  fresh third-party dependency resolution.
- The installed application now exposes a fail-closed `rles wfo` runner. It
  precharges Validation/Test budgets, binds score-only fold scalers, executes
  B0-B3 through the historical native engine, computes bootstrap/DSR/PBO when
  identifiable, performs final pre-Holdout retraining, and enforces the
  reviewer-approved one-shot Holdout ledger transition.
- WFO requires explicit native liquidation counts for every fold. The adapter
  now supplies them when simulation is enabled; a real WFO call passes this gate
  and blocks on the absent verified historical rule manifest.
- Extended real archive snapshot:
  `62239c22c3d069657ecb39ea109cffd1032a3dd039e3b46ee57db5fac0a77d21`.
  It spans 108 full UTC days (2023-12-30 through 2024-04-15), with 155,520
  each of OHLCV/mark/index events, zero missing or duplicate bars, and 323
  funding settlement/update pairs. All 329 source archives passed Binance
  `.CHECKSUM` verification.
- Full-range `market-audit` reports DQ-C for Train and evaluation with no missing
  minutes, duplicate facts, or conflicts. Mark/index/funding temporal coverage
  passes; the remaining findings are derived availability clocks and missing
  historical instrument rules.
- The WFO settlement policy now supports an explicit timestamp tolerance while
  retaining exact matching by default. The observed Binance source schedule
  passes at 4 ms tolerance; 37 of 273 settlements in the Jan 1–Apr 1 training
  window occur 1–4 ms after an 8-hour anchor.
- The actual 60/15/15 application WFO attempt now passes liquidation telemetry,
  loads the verified archive, and blocks on the missing historical rule manifest
  before fold generation or training.
- A one-day official Binance bookTicker companion dataset now contains 86,394
  last-per-second quote snapshots from 7,376,016 raw records. Its exchange ZIP
  SHA-256 is recorded, the 1 ms derived availability delay and sampling cadence
  are explicit, and every quote decodes through Nautilus `QuoteTick`. This adds
  genuine bid/ask prices and displayed sizes to replay. Quote mode now uses
  Nautilus native matching with displayed quote liquidity. A0 PPO/SAC/TD3/DDPG,
  A1 standalone ES and A4 E1→R1→E2 each completed config-driven training, bundle
  export/load and native quote replay on 2024-01-01 Binance data. All six
  reports show DQ-B, five Binance market sources, zero duplicate/conflict facts,
  and terminal flat; the four RL runs have 339–517 native fills. The executions
  used an explicitly scenario-only rule manifest; they validate route
  integration, not historical trading economics. Their reports distinguish
  market-source provenance from rule provenance.
- After quote execution, funding availability, visible-event indexing, config
  validation and reporting updates, the latest source suite reports **249
  passed and 4 CUDA-only skipped**; Ruff passes and MyPy reports no issues in
  97 source files. The previously verified wheel set
  predates these changes and must be rebuilt before release.

## Fail-closed real-data status

The expanded archive has enough calendar coverage for one 60/15/15 fold plus a
15-day Holdout, and no longer has minute or mark/index/funding gaps. It is still
DQ-C because the full dated ETHUSDT historical rule manifest is absent. Official
dated sources establish a 20 USDT minimum notional and the 0.40% maintenance
rate for positions up to 50,000 USDT from 2023-12-24 09:35 UTC. The current
60/15/15 config caps notional at 1,000 USDT, so this single tier is representable
by the current static Nautilus margin field. Tick size, quantity step, minimum
quantity, account-specific maker/taker rates, and source-hash provenance remain
open; larger exposure also needs full tiered margin execution.
The installed `validate` command was run and blocks on the absent verified
`historical_rules.json`. An actual WFO call passes liquidation telemetry, then
blocks on that rule manifest. The 360/15/15 horizon still needs more market history.

## Remaining development and acceptance work

1. Supply and verify authoritative historical rule documents, then create the
   normalized rule manifest and run the real native train/backtest path.
2. Reconcile native simulated-liquidation behavior against an independent
   expected-value golden and Binance's historical tiered margin rules.
3. Execute the application-level WFO runner on sufficient real data and add
   actual execution for the complete frozen 20-case stress protocol. The base
   WFO/selection/Test/baseline/statistics/retrain/Holdout runner is implemented;
   real evidence and stress replay are still absent.
4. Add application-level R1 checkpoint orchestration and stage-boundary resume,
   including replay journals, normalizer/RNG/seed schedules and durable budget
   reconciliation across attempts.
5. Complete CUDA ES checkpoint serialization and active GPU lease ownership, or
   keep the current fail-closed CPU-only restriction for ES.
6. Execute OS kill/OOM/disk-full/resource-limit recovery and three-repeat
   CPU/GPU/Hybrid performance tests.
7. Add independently expected real native fill/account golden traces and run the
   nightly/release gates. The existing synthetic contract tests must not be
   relabeled as real evidence.
