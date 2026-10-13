# rles-es-evotorch

EvoTorch-backed evolutionary search provider for Nautilus-RLES E1/E2 workflows.

The package exposes its provider through the `nautilus_rles.es` entry-point
group and consumes the shared `rles-contracts` interfaces. It supports PGPE
and CMA-ES, generation-boundary continuation, and a pickle-free native
checkpoint codec pinned to EvoTorch 0.6.1 on CPU. Recovery tests compare the
post-resume candidate stream with uninterrupted evolution. Checkpoints bind
actor, dataset, optimizer configuration, attempt, generation, and evaluation
count. Batched lane checkpoints also preserve the dispatched candidate/episode
request keys and their budget, preventing duplicate evaluation after resume.

This package does not certify strategy performance or provide production
market-data evidence. CUDA checkpoint/resume, external evaluator data cursors,
the production L3 payload factory, and SelectionLedger receipts remain
integration gates. The native codec
rejects other EvoTorch versions and does not load pickle.
