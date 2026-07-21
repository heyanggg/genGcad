# Experiment archive

This directory separates immutable SmartGen-GCAD experiment records from the active
`SmartGen/` runtime tree and the checked-in upstream baselines.

- `verified_candidates/`: high-scoring single-run candidates awaiting broader validation.
- `formal/`: accepted, provenance-complete formal single-seed experiments.
- `completed/`: technically complete runs awaiting research classification.
- `diagnostic/`: temporary debugging or calibration runs; remove after their conclusions are recorded.
- `failed/`: failed formal runs with enough evidence to diagnose the failure.
- `registry.json`: generated summary of every archived run.

Each run contains a tracked `README.md`, `manifest.json`, and `checksums.sha256`.
Its `artifacts/` directory is a local, self-contained snapshot and is intentionally
ignored by Git because model checkpoints are large. Back up the heavy artifacts using
Git LFS, a GitHub Release, or external storage before treating the archive as durable.

Complete generation+detection runs launched through `SmartGen/main.py` are archived
automatically under `completed/`. Use `--archive-status diagnostic` for an explicitly
diagnostic run, `--archive-status verified_candidates` for a pre-designated candidate,
`--archive-status formal` for an already accepted formal run, or `--no-archive` only
when no archive should be produced.

Useful archive commands, run from the repository root:

```bash
python SmartGen/archiving.py list
python SmartGen/archiving.py verify
python SmartGen/archiving.py promote <archive-id>
```

Only accepted `formal` archives are long-term local assets. Diagnostic artifacts are
disposable and should not be used as evidence for the reported formal results.
