# Codex agent file-based generation protocol

This backend is asynchronous file exchange, not a Codex API. The legacy `export` path remains for A1 compatibility. A2 uses `export-grouped-baseline`, which reads the official precomputed SPPC group files and writes deterministic, group-local requests containing the prompt SHA256, requested count, representative IDs/count/path/hash, source length summary, static metadata, schema 1.0, and a false-valued target-use declaration. Prompts are copied verbatim to `prompt_archive/`; no combined 305-representative prompt is created.

The active Codex agent authors each request independently. For A2, the explicit plan is passed to `materialize-authored`, a non-generative serializer that creates JSONL responses with matching request/group IDs, structured events, unique sequence IDs, and notes that no target behavior/labels or old synthetic data were used. It does not invent an API model ID, seed, token usage, or temperature.

`validate` rejects non-JSON/Markdown, missing/extra IDs, request/group mismatches, schema/type/empty/length violations, illegal device-action pairs, within/across-group duplicates, and wrong group counts. It retains failures and supports replacement by sequence ID; `replacement_mapping.json` is always materialized. It never compares against target behavior. `convert` deterministically converts validated structured events to SmartGen flat integer quadruples and proves the PKL is readable by original TOF.

```bash
python -m SmartGen.gcad_source.cli export-grouped-baseline --output OUT --experiment-id fr-spring-a2-r1 --dataset fr --source-context winter --context spring --threshold 0.918 --group-plan configs/generation_protocol/fr_spring_groups.json --target-metadata META.json --original-gss GSS.json --device-control DEVICE_CONTROL.json
python -m SmartGen.gcad_source.cli materialize-authored --directory OUT --plan OUT/codex_authored_plan.json
python -m SmartGen.gcad_source.cli validate --directory OUT
python -m SmartGen.gcad_source.cli convert --directory OUT --dataset fr
python -m SmartGen.gcad_source.cli diagnose-generation --directory OUT --dataset fr --original-gss GSS.json
python -m SmartGen.gcad_source.cli continue-pipeline --directory OUT --dataset fr --context spring --tof-epochs 10
python -m SmartGen.gcad_source.cli prepare-generated --generated OUT/tof/tof_sequences.pkl --dataset fr --context spring --output OUT/downstream_prepared --percentile 95.5 --epochs 15
python -m SmartGen.gcad_source.cli gate-reconstruction --directory OUT --diagnostics OUT/downstream_prepared/training_diagnostics.json
# Only after both gates and the freeze manifest pass:
python -m SmartGen.gcad_source.cli evaluate-prepared --prepared OUT/downstream_prepared --dataset fr --context spring
```

A1 outputs remain under `outputs/codex_generation/` and `experiment_artifacts/fr_spring/`. A2 is isolated under `outputs/codex_generation_v2/`; compact hashes and results are versioned under `experiment_artifacts/fr_spring_baseline_v2/`.
