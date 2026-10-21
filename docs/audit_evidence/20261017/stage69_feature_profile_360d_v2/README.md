# Stage69 360-day Train-inner feature profile

This is an engineering profile of feature construction over the half-open window
`[2024-12-01, 2025-11-26)` UTC. It is not policy training, candidate selection,
or evidence of profitability. Instrument assumptions are scenario-only.

The preregistration freezes the two isolated legacy v0.4.0 source snapshots,
their markers and import manifests, the rule assumption file, code identities,
funding schedule, and a 2 GiB minimum available-memory stop line. The runner
records both full-snapshot Train access scopes before decoding either source.
The project research registry is local and does not establish historical
completeness before registry activation.

To repeat the metadata-only preflight from the repository root:

```powershell
$env:PYTHONPATH = (Get-ChildItem packages -Directory | ForEach-Object { Join-Path $_.FullName 'src' }) -join ';'
.venv\Scripts\python.exe docs/audit_evidence/20261017/stage69_feature_profile_360d_v2/preflight.py
```

The original executed runner and frozen config are included here. Its preregistration
pins the original runner copy at the ignored `.tmp` path named in `code_sha256`;
the included runner is byte-identical. Re-running it requires the same local source
snapshots and Windows memory APIs. Its source window registration is idempotent.

`profile-result.json.sha256` and `preregistration.json.sha256` are raw-file SHA-256
receipts. The result reports feature-matrix and snapshot identities, build time,
process peak RSS, available-memory minimum, data-quality grade, and zero formal
evaluation queries.
