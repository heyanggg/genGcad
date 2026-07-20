# SmartGen + Source-only GCAD

This repository extends the official [SmartGen](https://github.com/horizonsinzqs/SmartGen) code with an optional source-only branch that extracts **GCAD-style predictive directional relations**. `main` remains the unmodified official baseline (`c2ed36c`); the baseline repair is on `baseline-protocol-v2`. The local reference implementation is `/home/heyang/projects/GCAD`, the official `Tc99m/GCAD` checkout. It was audited but not modified.

The added path is:

`TSS full source sequences → tensorizer → Mixer predictor → per-output gradients → asymmetric/stable relation → conservative GSS adapter → Codex agent file generation → original two-stage TOF → soft ranking → original downstream detector → final target evaluation`.

Training, relation extraction, prompting, generation, TOF, ranking, validation-threshold estimation, and hyperparameter selection use no target behavior data. Target normal and attack records enter only `evaluate-generated`, after the detector and generated-validation threshold are frozen. Static target device/action metadata and the textual context change remain allowed.

## Environment and quick start

Use `/home/heyang/miniconda3/bin/conda run -n smartguard_env`. The smoke run used Python 3.12.7 and PyTorch 2.13.0+cu130. CUDA was requested first but is genuinely unavailable: the installed NVIDIA driver reports CUDA 12.6 compatibility while this PyTorch build requires a newer driver, so the recorded experiment ran on CPU.

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

Detailed design, data roles, provenance, generation protocol, and audit material are under `docs/`. Large run products and checkpoints remain local under `outputs/`; compact, versioned evidence is under `experiment_artifacts/`.
