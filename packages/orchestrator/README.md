# rles-orchestrator

Coordinates the E1/R1/E2 hybrid workflow and records the lineage connecting
evolutionary search, reinforcement learning, and evaluation stages.

The package consumes shared `rles-contracts` data and workflow interfaces.
Production data, full application assembly, and research signoff remain
integration-gated.

## Train mechanism diagnostics

Use `rles diagnose-train --config diagnostic.json --bundle <trained-run>` to
replay the bundle's frozen Train window with its saved scaler, RL/ES actor and
soft-risk parameters. Keep `environment` identical to the training recipe and
set `output` to a new directory. The command ignores `evaluation`, performs no
training or scaler fitting, opens no research queries, and leaves the original
bundle and ledger unchanged. Its checksummed report and completion receipt use
`rles.train-diagnostic.v1`. It is exploratory evidence, not independent strategy
acceptance. Shared archive integrity checks may read the whole file; only the
Train window is scored.

`execution_attribution` groups requested, soft-risk, hard-risk and actual
exposures by native plan action/reason, alongside direction and participation
statistics. Missing risk stages remain unknown. SUBMIT indicates an order plan;
the final exposure gap can also include quantity rounding, pending orders,
fills and mark movement. Terminal settlement is reported separately from
pre-terminal policy exposure.

## Paired WFO cost-stress Validation

WFO runs a stress-mode replay by default for every trained candidate and fold,
using the same Validation interval, bundle, and frozen scaler with the
environment's declared stress latency, fill profile, and fee multiplier. The
fee multiplier affects positive maker/taker charges in the native fee model;
historical instrument-rule facts and maker rebates are not increased. Reports
include the resolved fee multiplier and execution-scenario hash. Each stress
replay is reserved as a separate Validation query before execution; its report
and receipt bind the bundle hash, fold interval, mode, and report hash. A candidate
must clear the frozen net-return, drawdown, turnover, liquidation, hard-risk,
and terminal-flat gates in every stress fold before it can reach Test. A
reserved stress replay without a complete verified report is never replayed on
resume. Set `wfo.cost_stress_validation` to `false` only for a protocol that
explicitly does not claim execution-cost robustness.

## Application checkpoint recovery

The public `train` application accepts `resume_checkpoint` for an existing
incomplete run. For R1, point it at a completed `R1-checkpoints/step-*`
boundary. For E1, point it at the attempt directory inside
`E1/generation-*/<attempt-id>` (or the generation directory itself). For E2,
use the corresponding `E2/generation-*/<attempt-id>` checkpoint. Recovery
requires the original output directory, matching frozen recipe/RunSpec and
valid checkpoint integrity. R1 recovery is available on A0/A2/A3/A4; E1
recovery is available on A1/A3/A4; E2 recovery is available on A4. E2
recovery verifies that the full-budget R1 checkpoint re-exports the same
actor hash before reusing the existing L4 inference artifact. ES recovery
resumes from completed generation boundaries and reconciles the original L6
reservations.

## E1/E2 stage executor

Install the `native-es-runtime` extra to use
`rles_orchestrator.es_stage.ESStageExecutor`; add `native-es-worker` when
using the bundled Native Gym worker below. Supply an `EvolutionConfig`,
ActorGenomeSpec (E1) or SoftRiskSpace (E2), frozen worker factory/config,
Train-inner episode IDs, L6 `ResourceQuota` and `SelectionLedger`, and a
checkpoint directory. Use the executor as the callback to
`ProviderBackedStageRunner`; `Orchestrator` then verifies stage usage and
persists the receipt.

## R1 stage executor

Install the `native-rl-runtime` extra to use
`rles_orchestrator.rl_stage.RLStageExecutor` as an R1 callback to
`ProviderBackedStageRunner`. Supply the L4 `RLConfig`, a public environment
factory, a gate authorizer, a bounded training step count, fixed raw
observations for model export, and an inference artifact directory. For an
E1→R1 route, also supply the attested E1 `ActorGenomeSpec` and genome plus
Train-inner observations for transfer. The executor checks the frozen run
and stage budget, performs L4 direct transfer or L5 Train-inner distillation,
trains through the independent L4 provider, exports its L0 inference actor,
checks prediction parity, and closes the provider. A pure R1 route omits
the E1 inputs.

The executor checks stage caps and E2's frozen R1 actor before dispatch. For
the bundled Native worker, it verifies the L4 inference artifact before
reserving L6 candidate requests. It
reserves every candidate × episode request in L6, runs the actual EvoTorch
provider through spawn workers, and returns a `ProviderStageOutput` with
measured evaluations, lane requests, fitness evidence, and artifact identity.
It saves each completed generation in a separate immutable checkpoint
directory. A new executor with `resume=True` loads the latest completed
generation, reconciles L6 reservations, and finishes the stage. No RL or ES
algorithm is copied into this workflow package.

## Native ES worker adapter

Install the `native-es-worker` extra to use
`rles_orchestrator.native_es_worker:make_native_es_session` as the L6
`CPUWorkerPool.factory_ref`. The application supplies an installed public
`env_factory_ref` callable. Each call receives `env_config`, the L5
`candidate_context`, and E2 `soft_risk_multipliers`, then returns a fresh
`NautilusBarrierEnv` with its own reviewed L3 episode and Native Engine.

The JSON worker configuration fixes `candidate_kind`, `actor_spec`,
`actor_signature`, `transition_hash`, `episode_steps`, `env_factory_ref`, and
`env_config`. E2 additionally fixes `risk_dimensions`,
`actor_inference="l4_inference"`, an absolute `frozen_actor_artifact` path,
and its `frozen_actor_hash` from the R1 L0 export. L6 adds `worker_index`.
The adapter verifies the L5 request hash, frozen candidate schema, ordered
observation names, transition identity, and terminal settlement. Its output
contains native equity, raw
economic log returns, drawdowns, intervention totals, and bound evidence for
`WorkerLaneDispatcher` and `FitnessSpec`; it does not use shaped Gym reward.
E2 evaluates the exported L4 policy, including its own action mapping and
frozen observation normalization, rather than decoding trained weights as
an L5 canonical actor.

The worker factory and its configuration are included in the L6 lane runtime
signature consumed by L5 checkpoint identity. Changing either requires a new
attempt. The application remains responsible for sourcing frozen Train-inner
market data and building a separate native account for every candidate.
