# Experiment protocol

A5 changes only generation-time support allocation and deterministic candidate selection. Its 160 pool size, group quotas, uniform support margins, deletion ordering, and all downstream gates must be frozen before the first candidate is authored.

The first cell is FR winter→spring, chosen before target evaluation. Frozen v1 source settings were device-action/binary/3-hour tensorization, history 4, 3 Mixer layers, hidden 96, batch 64, Adam 1e-3, weight decay 1e-5, maximum 30 epochs, patience 6, gradient clip 1.0. Full-source seeds were 2024–2026 plus partition replicates 3101/3102. V1 GCAD settings and results are preserved as smoke evidence, not extended here.

Baseline protocol v2 freezes generation at 137 samples allocated across the 15 official SPPC source groups (`8,6,12,10,7,6,6,8,8,29,9,9,6,7,6`). Group-local length bounds come from source q10/q90 only. Original TOF uses 10 epochs per fit. Downstream uses seed 2024, deterministic whole-sequence 80/20 splitting, 15 epochs, and generated-validation percentile 95.5. Exact duplicate sequences cannot cross the split.

Groups are:

- A: baseline prompt/GSS, baseline generation and TOF, uniform training.
- B: GCAD-GSS prompt, GCAD generation and TOF, uniform training.
- C: exactly A's TOF PKL plus soft relation weights.
- D: exactly B's TOF PKL plus soft relation weights.
- E/F: random-directed and symmetric control interfaces implemented, full generation/evaluation deferred.

For A2, validation/materialization, source-only generation diagnostics, TOF, detector training, threshold creation, reconstruction gating, and the pre-target checksum manifest were complete before any target file was opened. `evaluate-prepared` then performed one final target evaluation. Its metrics cannot select parameters, thresholds, gates, or batches. A2 did not improve A1, so this branch does not create or run GCAD representation v2 and does not expand the other cells.

After A2 was closed, `source_semantic_v1` was frozen for future replicates. It calibrates action/transition support by leave-one-source-day-out coverage and checks global support, per-group anchors, zero-anchor share, and vocabulary expansion. Because it was introduced after A2 target metrics were viewed, its A2 diagnosis is explicitly post hoc. Future TOF execution requires both the original distribution gate and this semantic gate; passing them does not authorize a target evaluation by itself.

Run tests with `/home/heyang/miniconda3/bin/conda run -n smartguard_env python -m pytest -q`. Ranking now uses an identical full-coverage DataLoader plus bounded per-sample weighted loss; uniform signals bypass weighting exactly. Invalid semantic channels such as `None:location` fail immediately.
# A4 outcome

The A4 replicate-4 protocol stopped at pre-TOF `source_semantic_v2`. Its failure does not authorize changes to the same replicate; all target-data and GCAD stages remain blocked.
