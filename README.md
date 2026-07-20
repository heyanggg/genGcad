# SmartGen + Source-only GCAD

GCAD representation v2 has now completed its pure-source decision. It repaired v1's representation defects (579 formal event-position windows, 39 legal channels, zero source-split leakage), but the selected Mixer lost to n-gram baselines on all three fixed seeds. The formal source prediction gate failed, so no stable relation, GSS fusion, B5 generation, or target evaluation was run. The current recommendation is to terminate the GCAD directional-relation route; see [gcad_v2_final_decision.md](docs/gcad_v2_final_decision.md).

Current A5 status: replicate-5 directly authored 160 Codex GPT-5.6 candidates, selected a valid 137 subset, passed both semantic and all fixed-split checks before and after formal CPU TOF, then stopped at the unchanged `reconstruction_health_v1` gate because high-loss samples dominated and thresholds were unstable across seeds. It never accessed target behavior or ran final evaluation, GCAD, or Ranking. See [replicate5_results.md](docs/replicate5_results.md).

A5 was preregistered as a 160-candidate, support-aware Codex-agent pool reduced deterministically to the unchanged 137 group quotas. Python selected but never authored or edited events; all v4 semantic, split, reconstruction, and 95.5% threshold rules remained unchanged.

Replicate-3 is frozen and reclassified as a Programmatic Source-Constrained Composition Stress Test; it is not
an LLM baseline. Formal GPT-5.6-via-Codex generation work proceeds only on
`codex56-generation-protocol-v4` with Python prohibited from constructing event content. See
`docs/replicate3_programmatic_reclassification.md` and `docs/codex_gpt56_generation_definition.md`.

This repository extends the official [SmartGen](https://github.com/horizonsinzqs/SmartGen) code with an optional source-only branch that extracts **GCAD-style predictive directional relations**. `main` remains the unmodified official baseline (`c2ed36c`); the baseline repair is on `baseline-protocol-v2`. The local reference implementation is `/home/heyang/projects/GCAD`, the official `Tc99m/GCAD` checkout. It was audited but not modified.

The guarded research path is:

`pre-TSS source sequences → canonical representation → leakage-safe source prediction gate → [blocked unless passed] per-output gradients → stable relation → conservative GSS adapter → Codex agent generation → original two-stage TOF → original downstream detector → paired final evaluation`.

Training, relation extraction, prompting, generation, TOF, ranking, validation-threshold estimation, and hyperparameter selection use no target behavior data. Target normal and attack records enter only `evaluate-generated`, after the detector and generated-validation threshold are frozen. Static target device/action metadata and the textual context change remain allowed.

## Environment and quick start

Use `/home/heyang/miniconda3/bin/conda run -n smartguard_env`. The v5 request context required the formal experiment to run on CPU. A CUDA device unexpectedly became visible during execution; that non-formal attempt is retained separately, while the formal TOF and reconstruction runs explicitly hid CUDA and report CPU execution.

```bash
/home/heyang/miniconda3/bin/conda run -n smartguard_env python -m pytest -q
bash scripts/run_source_gcad_smartgen.sh fr winter spring configs/gcad_source/fr.yaml
# after Codex-authored JSONL responses exist
bash scripts/continue_codex_cell.sh fr spring 95.5
```

The unified CLI additionally exposes grouped baseline export, authored-response materialization, generation diagnostics, separate detector preparation, reconstruction gating, and final prepared evaluation. See [the generation audit](docs/smartgen_generation_protocol_audit.md), [A2 design](docs/codex_baseline_v2_design.md), and [A2 result](docs/baseline_v2_results.md).

To disable the extension, set `gcad_source.enabled: false` and use the baseline prompt path. In this mode the adapter returns the baseline prompt byte-for-byte, does not load a GCAD checkpoint or relation artifact, and leaves SmartGen GSS, TOF, paths, and downstream entry points unchanged.

The historical FR winter→spring A1 smoke generated 137 baseline and 137 GCAD-GSS sequences without an API, API key, target behavior, or old synthetic data. Its baseline F1 was 0.7733 and the three v1 enhancement paths were 0.6667, but ranking was confounded by replacement sampling and GCAD admitted an invalid `None:location` channel.

The A2 repair uses the 15 official precomputed SPPC source groups, 16 independent offline Codex-file requests, source-derived length limits, source/internal generation gates, deterministic duplicate-safe splitting, and a detector frozen before one target evaluation. It eliminated illegal actions and action-template collapse, but did **not** improve performance: precision 0.7647, recall 0.4432, F1 0.5612, FPR 0.1364, threshold 4.2920. This negative result was not used to regenerate or retune A2. GCAD representation v2 remains paused.

A post-A2 source-only audit found the missing control: only 36.79% of A2 events and 3.08% of adjacent transitions are supported by FR winter behavior, while 38/137 sequences have no source action anchor. `source_semantic_v1` now calibrates coverage by leave-one-source-day-out validation, embeds group anchors in future Prompts, and blocks TOF on failure. See [the semantic-gate design](docs/source_semantic_gate.md).

The prospective source-semantic replicate-2 generated 137 valid records but stopped at the original distribution gate because three short sequences independently matched representatives from other SPPC groups. No post-raw repair, regeneration, source-semantic scoring, TOF, downstream training, or target evaluation followed. See [the frozen failure report](docs/prospective_replicate_2_source_gate_failure.md) and [the unused reconstruction-health policy](docs/reconstruction_health_v1.md).

Detailed design, data roles, provenance, generation protocol, and audit material are under `docs/`. Large run products and checkpoints remain local under `outputs/`; compact, versioned evidence is under `experiment_artifacts/`.
