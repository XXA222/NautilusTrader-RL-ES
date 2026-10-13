# rles-compute-runtime

The runtime supplies CPU worker isolation, attempt resource quotas and exclusive
CUDA learner ownership. SQLite quotas reserve budgets; OS memory enforcement is
provided separately by Windows Job objects for configured worker pools.

## Explicit CPU quota recovery

Inspect a run without changing its quota database:

```powershell
python -m rles_compute_runtime.recovery inspect --database <run>/quota.sqlite
```

For a reviewed lease with an exited coordinator, provide its exact descriptor:

```powershell
python -m rles_compute_runtime.recovery recover --database <run>/quota.sqlite --token <token> --owner <owner> --cpu-cores <cpu> --ram-bytes <ram> --reason "owned coordinator hard exit"
```

Recovery is supported for new Windows `CPUWorkerPool` leases with
`enforce_memory_limit=True` after complete startup. The pool records the original
coordinator's machine/session, PID and creation time plus token-specific named
Jobs before admitting factory work. It seals the enrollment after all workers
report READY. Recovery checks that exact owner has exited and every registered
Job is absent or empty. It releases the exact CPU lease and records a durable
`rles.cpu-lease-recovery.v1` receipt in one transaction. Concurrent or repeated
recovery cannot release the same lease twice.

`ready` in inspection means startup was sealed; it does not prove the owner or
tree has exited. Active or unknown owners/trees, foreign machine/session,
incorrect descriptors, incomplete startup, legacy/unmanaged pools and GPU leases
are refused. Recovery does not remove GPU locks, refund research reservations,
resume training, or validate checkpoints. In-process learner and coordinator
memory limits remain separate work.

The Job tree assumes trusted factories use standard Windows child-process
creation without external service escapes. No breakaway flags are enabled.
The runtime never performs recovery by terminating a stale-looking PID.
