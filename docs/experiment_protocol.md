# Experiment protocol

The first cell is FR winter→spring, chosen before target evaluation. Frozen source settings: device-action/binary/3-hour tensorization, history 4, 3 Mixer layers, hidden 96, batch 64, Adam 1e-3, weight decay 1e-5, maximum 30 epochs, patience 6, gradient clip 1.0. Full-source seeds are 2024–2026. Two additional 80% disjoint partition replicates use seeds 3101/3102. Relation thresholds are edge 0.01/top-k 5, occurrence 0.6, direction consistency 0.6, stable score 0.08. Fusion is `rerank_existing`, alpha 0.2, no new edges. Generation is 137 samples, batches 20/17. Original TOF uses 10 epochs per fit. Downstream uses seed 2024, 15 epochs, and generated-validation percentile 95.5.

Groups are:

- A: baseline prompt/GSS, baseline generation and TOF, uniform training.
- B: GCAD-GSS prompt, GCAD generation and TOF, uniform training.
- C: exactly A's TOF PKL plus soft relation weights.
- D: exactly B's TOF PKL plus soft relation weights.
- E/F: random-directed and symmetric control interfaces implemented, full generation/evaluation deferred.

All source parameters, both generation sets, TOF results, and detector protocol were frozen before any target file was opened. Final metrics cannot select parameters or batches. This is a one-replicate smoke experiment, not inferential evidence; expansion should add Codex generation replicates and all nine cells without changing the frozen boundary.

Run tests with `/home/heyang/miniconda3/bin/conda run -n smartguard_env python -m pytest -q`. Execute individual stages with `SmartGen.gcad_source.cli`; `scripts/run_source_gcad_smartgen.sh` covers tensorization, three seed trainings, and relation extraction.
