# SmartGen + Source-only GCAD

This repository extends the official [SmartGen](https://github.com/horizonsinzqs/SmartGen) code with an optional source-only branch that extracts **GCAD-style predictive directional relations**. `main` remains the unmodified official baseline (`c2ed36c`); development is on `gcad-integration`. The local reference implementation is `/home/heyang/projects/GCAD`, the official `Tc99m/GCAD` checkout. It was audited but not modified.

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

The unified CLI exposes `tensorize`, `train`, `extract-relations`, `build-stable-relation`, `fuse-gss`, `build-prompts`, `export`, `validate`, `convert`, `continue-pipeline`, `evaluate-generated`, and `build-mechanism-control`. See [the runbook](docs/gcad_integration_runbook.md) for exact commands and [the experiment report](experiment_artifacts/fr_spring/report.md) for real results.

To disable the extension, set `gcad_source.enabled: false` and use the baseline prompt path. In this mode the adapter returns the baseline prompt byte-for-byte, does not load a GCAD checkpoint or relation artifact, and leaves SmartGen GSS, TOF, paths, and downstream entry points unchanged.

The first completed cell is FR winter→spring. It generated 137 new baseline and 137 new GCAD-GSS sequences in seven batches each without an API, API key, target behavior, or old synthetic data. The enhancement did **not** improve the downstream result: baseline F1 was 0.7733; GCAD-GSS, ranking-only, and both were 0.6667. The Mixer also lost to Markov/n-gram validation loss. These negative findings and missing official SPPC checkpoints are documented in [known limitations](docs/known_limitations.md).

Detailed design, data roles, provenance, generation protocol, and audit material are under `docs/`. Large run products and checkpoints remain local under `outputs/`; compact, versioned evidence is under `experiment_artifacts/`.
