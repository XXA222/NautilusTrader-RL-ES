# Stage69 Train-inner Episode Reset profile

This profile measures cold and warm Episode Reset for 60- and 360-day windows
inside the frozen `[2024-12-01, 2025-11-26)` UTC Train-inner period. It uses
the same two physically isolated v0.4.0 market snapshots as the Stage69 360-day
feature profile. Both complete source scopes are registered before either
Parquet payload is decoded; the legacy reader then verifies every footer bound
and the complete file hash. The metadata-only preflight found 117 row groups,
1,907,766 events, 1,080 scheduled Funding settlements, and no formal Test or
Holdout interval overlap.

Run the metadata-only preflight from the workspace root:

```powershell
$env:PYTHONPATH = (Get-ChildItem packages -Directory | ForEach-Object { Join-Path $_.FullName 'src' }) -join ';'
.venv\Scripts\python.exe docs\audit_evidence\20261017\stage69_episode_reset_360d_v1\preflight.py
```

The executed profile command was:

```powershell
$env:PYTHONPATH = ((Get-ChildItem packages -Directory | ForEach-Object { Join-Path $_.FullName 'src' }) -join ';') + ';qa'
.venv\Scripts\python.exe qa\profile_stage69_episode_reset_20261017.py --preregistration docs\audit_evidence\20261017\stage69_episode_reset_360d_v1\preregistration.json --config docs\audit_evidence\20261017\stage69_episode_reset_360d_v1\config.json --output .tmp\stage69-episode-reset-20261017-v1
```

Results: 60-day cold/warm Reset **3.179 s / 0.0098 s**; 360-day **21.538 s /
0.0838 s**. The observation and initial native account fingerprint matched
across same-seed cold/warm resets and across the two windows. Peak process RSS
was 4,519,305,216 bytes; minimum available memory at the profile's phase
checkpoints was 13,250,097,152 bytes, above the frozen 2 GiB stop line. The 360-day factory
created 518,400 row frames and barrier frames. See `profile-result.json` for
the per-window measurements and hashes.

Rules are scenario-only. The profile performs no policy training, model
selection, formal query, or profitability evaluation; its result is limited to
data preparation, Reset reproducibility, and resource behavior. The existing
Stage69 feature-profile evidence is the source identity and merged snapshot
reference. Preregistration and result checksums are stored beside this file.
