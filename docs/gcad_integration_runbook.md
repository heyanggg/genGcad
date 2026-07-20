# Integration runbook

Always run from `/home/heyang/projects/SmartGen_GCAD` on `gcad-integration` with `/home/heyang/miniconda3/bin/conda run -n smartguard_env python`.

`run_source_gcad_smartgen.sh` runs the complete pre-generation source pipeline and exports both request sets. After the Codex agent writes raw responses, `continue_codex_cell.sh` validates, converts, runs original TOF and writes rankings. It deliberately requires `RUN_FINAL_EVALUATION=1` before opening target data for A/B/C/D.

```bash
# Source branch through per-seed relations
bash scripts/run_source_gcad_smartgen.sh fr winter spring configs/gcad_source/fr.yaml

# Post-response path; target data remain unopened
bash scripts/continue_codex_cell.sh fr spring 95.5

# Final evaluation only after configuration is frozen
RUN_FINAL_EVALUATION=1 bash scripts/continue_codex_cell.sh fr spring 95.5

# Stable relation (five matrices in the completed experiment)
python -m SmartGen.gcad_source.cli build-stable-relation --matrices M1.npy M2.npy M3.npy M4.npy M5.npy --primary-lags L1.npy L2.npy L3.npy L4.npy L5.npy --vocabulary outputs/gcad_source/fr/spring/tensor/channel_vocabulary.json --output outputs/gcad_source/fr/spring/relations/stable --occurrence-threshold 0.6 --direction-consistency-threshold 0.6 --stability-threshold 0.08 --seeds 2024 2025 2026 3101 3102 --split-ids full full full partition_0 partition_1

# Conservative fusion and prompts
python -m SmartGen.gcad_source.cli fuse-gss --original-gss SmartGen/IoT_data/fr/winter/action_transitions.json --stable-relation outputs/gcad_source/fr/spring/relations/stable/stable_relation.json --target-metadata outputs/gcad_source/fr/spring/target_static_metadata.json --output outputs/gcad_source/fr/spring/fusion --alpha 0.2 --mode rerank_existing
python -m SmartGen.gcad_source.cli build-prompts --dataset fr --source-context winter --target-context spring --threshold 0.918 --stable-relation outputs/gcad_source/fr/spring/relations/stable/stable_relation.json --fused-gss outputs/gcad_source/fr/spring/fusion/fused_gss.json --output outputs/gcad_source/fr/spring/prompts

# E/F control artifact interfaces
python -m SmartGen.gcad_source.cli build-mechanism-control --relation outputs/gcad_source/fr/spring/relations/stable/stable_relation.json --output outputs/gcad_source/fr/spring/controls/random.json --mode random_directed --seed 2024
python -m SmartGen.gcad_source.cli build-mechanism-control --relation outputs/gcad_source/fr/spring/relations/stable/stable_relation.json --output outputs/gcad_source/fr/spring/controls/symmetric.json --mode symmetric

# Final evaluation; omit --ranking for A/B, add it for C/D
python -m SmartGen.gcad_source.cli evaluate-generated --generated TOF.pkl --ranking sequence_ranking.json --dataset fr --context spring --output OUT --percentile 95.5 --epochs 15
```

Replace the bare `python` commands above with the full conda prefix in formal runs. Do not run final evaluation until every source-only setting is frozen. For the remaining cells use mappings FR/SP/US × winter→spring, daytime→night, single→multiple and the dataset config/official percentile; do not reuse FR target feedback.
