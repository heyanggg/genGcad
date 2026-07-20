# Codex agent file-based generation protocol

This backend is asynchronous file exchange, not a Codex API. `export` writes deterministic requests containing the complete prompt, prompt SHA256, requested count, source count, static metadata, artifact paths, schema 1.0, and a false-valued target-use declaration. Prompts are copied verbatim to `prompt_archive/`. Counts are split into batches of 10–25; the first cell used six batches of 20 plus one of 17.

The active Codex agent reads each request and writes JSONL responses with matching IDs, method, batch, structured events, unique sequence IDs, and explicit notes that no target behavior/labels or old synthetic data were used. It does not invent an API model ID, seed, token usage, or temperature.

`validate` rejects non-JSON/Markdown, missing/extra IDs, schema/type/empty/length violations, illegal device-action pairs, within/across-batch duplicates, and direct source duplicates. It retains failures and supports replacement by sequence ID; `replacement_mapping.json` is always materialized. It never compares against target behavior. `convert` deterministically converts validated structured events to SmartGen flat integer quadruples and proves the PKL is readable by original TOF.

```bash
python -m SmartGen.gcad_source.cli export --output OUT --experiment-id fr-spring-r1 --dataset fr --context spring --method baseline --replicate 1 --prompt PROMPT --count 137 --batch-size 20 --source-sequence-count 1728 --context-description CONTEXT.json --target-metadata META.json --original-gss GSS.json
# Codex agent creates OUT/generation_responses_raw.jsonl
python -m SmartGen.gcad_source.cli validate --directory OUT
python -m SmartGen.gcad_source.cli convert --directory OUT --dataset fr
python -m SmartGen.gcad_source.cli continue-pipeline --directory OUT --dataset fr --context spring --tof-epochs 10 --stable-relation STABLE.json
```

Formal outputs and checksums for the completed run are under `outputs/codex_generation/fr/spring/{baseline,gcad_gss}/replicate_1/`; compact hashes are versioned in `experiment_artifacts/fr_spring/summary.json`.
