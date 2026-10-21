# TD3 fixed actor: 1x vs 10x cap development diagnostic

## Result

The frozen TD3 actor was replayed over the saved 2024-01-01 12:00 UTC to
2024-01-02 00:01 UTC window with a $10,000 starting balance and a $100,000
risk budget in both scenarios. The account leverage cap was 1x or 10x; the
actor, scaler, features, market snapshot, execution recipe, seed, fees, and
other risk settings were held fixed.

| Metric | 1x cap | 10x cap |
|---|---:|---:|
| Net return | -26.3994% | -37.3986% |
| Maximum drawdown | 26.3994% | 37.3986% |
| Total fees | 2,618.22 USDT | 3,935.81 USDT |
| Funding | -3.48757 USDT | 0 USDT |
| Slippage cost reported by engine | 18.05988 USDT | -13.22023 USDT |
| Gross return before reported costs | -0.0018% | +1.8273% |
| Turnover ratio | 608.65 | 988.78 |
| Peak observed gross leverage | 1.003x | 2.111x |
| Liquidation events | 0 | 0 |

The 10x cap was not fully utilized: observed exposure peaked at 2.111x. The
10x result therefore describes this actor under a 10x account cap, not a
10x continuously deployed position. Both scenarios had 719 return observations.
Fees dominated the small gross result, and both scenarios lost substantially.

## Funding reconciliation

The saved evaluation interval contains two scheduled settlements. At
2024-01-01 16:00 UTC, the funding rate was +0.00043011 and the mark was
2316.72581439 USDT. In the 1x run the actor held +3.500 ETH, so the linear
funding formula gives -3.48756429 USDT; the native account ledger charged
-3.48757059 USDT, a residual of -0.00000630 USDT. In the 10x-cap run the
actor was flat at that settlement, so no funding was charged. Both runs were
flat at 2024-01-02 00:00 UTC (rate +0.00053505), so the second settlement
also charged zero. Positive funding rates debit longs in this replay.

The first attempt failed closed because the scenario rule file omitted the
reduce-only minimum-notional exemption. The follow-up sensitivity explicitly
enabled that rule. Binance's current API documentation describes the
MIN_NOTIONAL exception for reduce-only orders, but that current documentation
does not establish the historical 2024 rule. The sensitivity is therefore
still classified as scenario-only.

## Limits

- This saved Validation window had already been used. These are repeated,
  non-independent development diagnostics, not an independent performance
  validation or a strategy promotion result.
- The actor was trained with a 1x cap and a 1,000 USDT risk budget. The paired
  replay uses a 100,000 USDT risk budget and is out of training distribution.
- The margin schedule is explicitly scenario-only. Fees, best-quote fills,
  and zero synthetic slippage assumptions do not certify historical account
  fees, exchange tiered margin, liquidation, or market impact.
- No 10x result should be inferred from the 10x cap alone: realized gross
  leverage peaked at 2.111x and the actor was flat at both funding settlements.

## Reproduction evidence

- Frozen actor SHA-256: `1aa440a94435d3546d221170cb944fc727bf78c42656f01aac891bda23d216fa`.
- Corrected run registration: [registration_r3.json](registration_r3.json).
- Result artifact: [summary_r3.json](summary_r3.json), SHA-256
  `6e6b13d85128279f2fef6cc2e3b43edd3d08cfcc84433090aa1165bed9e4ea7d`.
- The r3 runner uses signed fills through each settlement timestamp and the
  latest prior mark to independently reconstruct the funding calculation.
- Current Binance API reference for the reduce-only notional exception:
  [Error Codes](https://developers.binance.com/en/docs/products/derivatives-trading-portfolio-margin/error-code).

## Current-code replay after fee-aware cap (r4)

The same frozen actor and already-used Validation interval were replayed again
after adding the modeled transaction-cost reserve to the leverage guard. The
paired result remains a development-only sensitivity: the 1x run returned
-26.3994% with 2,618.21 USDT fees and -3.48757 USDT Funding; the 10x-cap run
returned -37.3986% with 3,935.81 USDT fees and zero Funding. Peak observed gross
leverage was 1.003x and 2.111x respectively, so this actor did not use the 10x
cap. Fee reserve changed the 1x fee/slippage figures slightly; it did not turn
this out-of-distribution fixed-actor replay into independent evidence.

- Current-code registration: [registration_r4.json](registration_r4.json).
- Current-code result: [summary_r4.json](summary_r4.json), SHA-256
  `cbf93bc6b9564f4424fa69fb87c56be34018be9efe20ec6235ecc57d3b7776df`.
- Full project test run after the RouteB removal and leverage-cost change:
  **783 passed, 5 CUDA-only skipped, 53 dependency warnings**.
