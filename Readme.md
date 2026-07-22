# Readme

This document contains part of the source code for the paper *"SmartGen: Synthesizing Context-Aware User Behavior Data for Adaptive Smart Home Intelligence"*. It includes four functional modules (TSS, SSC, GSS, TOF), data synthesis procedures, testing on two smart home tasks, parameter experiments, and ablation studies. Please note the following:

1. In the code, **SPPC** refers to **SSC**. SPPC is an earlier naming version that was not changed due to the established workflow.
2. The **SmartGen** folder contains the implementations of the four functional modules as well as the data synthesis system. The other six folders correspond to various experimental setups.
3. Each folder includes a `main.py` file, which serves as the entry point for running the corresponding experiment. The CSV files in the `results` directory contain the recorded outcomes of these experiments.
4. The original code did not include a usable large-language-model connection. This branch invokes GPT-5.6 through the locally authenticated Codex CLI.

See [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) for the active/legacy boundary,
runtime directory ownership, and the immutable experiment archive layout.

Recommended compression thresholds in SmartGen:
| Dataset | Original Context | New Context | Compression Threshold | Anomaly Detection Percentage |
| ------- | ---------------- | ----------- | --------------------- | ---------------------------- |
|         | winter           | spring      | 0.918                 | 95.5                         |
| FR      | daytime          | night       | 0.92                  | 95                           |
|         | single           | multiple    | 0.915                 | 99                           |
|         | winter           | spring      | 0.915                 | 95                           |
| SP      | daytime          | night       | 0.917                 | 95                           |
|         | single           | multiple    | 0.915                 | 99                           |
|         | winter           | spring      | 0.905                 | 95                           |
| US      | daytime          | night       | 0.919                 | 93                           |
|         | single           | multiple    | 0.913                 | 99                           |

## SmartGen + GCAD + Codex GPT-5.6

The generation path keeps the original TSS, SSC/SPPC, GSS, and TOF behavior. The GCAD branch reads the complete `split_trn.pkl` produced by TSS before SSC compression. Because SmartGen stores event sequences instead of equal-interval continuous sensor vectors, this integration is explicitly described as **GCAD-derived** rather than as proof of real-world causality.

The adapted GCAD stages are:

1. Encode behavior positions as multichannel one-hot history and target tensors.
2. Train a TSMixer-style multichannel predictor with per-channel MSE and early stopping.
3. On held-out source sequences, backpropagate each target channel separately to build the complete `source action × target action × lag` gradient tensor.
4. Integrate lag scores, apply `max(0, A - Aᵀ)`, and sparsify the graph.
5. Repeat with three model seeds and retain only relationships stable across at least two seeds.

The predictor must outperform a held-out frequency baseline. If the data are too small, validation fails, or no edge is stable, GCAD writes a disabled artifact with an empty relationship list; SmartGen then uses a byte-equivalent copy of the original prompt without any GCAD text. GCAD guidance is appended only when the artifact is ready and contains stable relationships. A successful relationship records its raw and normalized strength, lag, support, and seed stability. GCAD anomaly scoring is intentionally not included.

The GCAD/Codex integration is concentrated in three files:

- `SmartGen/gcad.py`: prediction, gradient relationship discovery, and graph sparsification.
- `SmartGen/codex_backend.py`: direct non-interactive Codex CLI invocation using `gpt-5.6-sol`.
- `SmartGen/experiment.py`: validated run configuration, run IDs, atomic writes, and resumable manifests.

The active experiment layout is:

```text
SmartGen/
├── main.py                 # The only SmartGen + GCAD experiment entry point
├── split.py / dayse.py     # Original TSS and day grouping
├── baseline2.py / sppc.py  # Original SSC/SPPC training and selection
├── text_translation_matrix.py  # Original GSS transition guidance
├── gcad.py                 # Additional lagged directional guidance
├── codex_backend.py        # Codex GPT-5.6 generation backend
├── extract.py / transnumber.py / security_check.py  # Original TOF path
├── IoT_data/               # Source data and per-category generation responses
├── artifacts/gcad/         # Rebuildable GCAD artifacts (local, ignored)
├── runs/                   # Manifest plus exact prompts/responses (local, ignored)
├── filter_data/            # Merged numeric and final TOF datasets
└── anomaly_runs/           # Detector checkpoint, split, and metrics per seed
```

The other top-level experiment folders are retained as upstream SmartGen baselines and studies. They are not imported by the active `SmartGen/main.py` path.

Use the `smartguard_env` virtual environment and run from `SmartGen/`:

```bash
conda activate smartguard_env
codex login status
cd SmartGen
python main.py --need_generate True --model gpt-5.6-sol
```

The runner can also be launched from the repository root. Every generation replicate must have a distinct `--run-id`; it becomes part of all generated filenames so later runs cannot overwrite earlier results. Prompt/response progress is saved under `SmartGen/runs/`, and valid category responses are reused after interruption unless `--no-resume` is passed:

```bash
conda activate smartguard_env
python SmartGen/main.py \
  --need-generate True \
  --need-test True \
  --dataset fr \
  --ori-env winter \
  --new-env spring \
  --threshold 0.918 \
  --run-id run1 \
  --experiment-seed 2024 \
  --prompt-profile environment-aware \
  --codex-reasoning-effort none
```

The artifact label for this example is `gpt-5.6-sol__run1`. GCAD still uses its three internal stability seeds, while `--experiment-seed` controls SSC, TOF, the GCAD holdout split, and anomaly-detector training. The anomaly detector writes its split, checkpoint and `metrics.json` under `SmartGen/anomaly_runs/` instead of overwriting target-domain source data.

By default, GCAD writes `SmartGen/artifacts/gcad/<dataset>/<original_environment>/gcad_hints.json`. The artifact includes a source-data hash and extraction configuration, so an unchanged result is reused instead of retrained. Pass `--gcad-force` to rebuild it. The original `action_transitions.json` is only read and is never reranked or overwritten by GCAD. Only the compact relationship list—not training diagnostics—is sent to the model, and it is labeled as soft predictive guidance.

GCAD prompt participation is explicit: `--gcad-mode auto` uses stable relationships when available, `--gcad-mode off` extracts/records GCAD but suppresses its prompt guidance for ablation, and `--gcad-mode require` stops the run unless stable relationships are available. The complete graph remains archived, while prompt guidance is balanced to at most 12 relationships and at most 4 relationships per target action by default. The selected mode, graph/prompt counts, and whether guidance was actually enabled are recorded in both the run manifest and experiment archive.

The default `environment-aware` prompt profile adds explicit target-environment and sequence-shape constraints while retaining the full upstream prompt. Night-sequence shape bounds were calibrated from the original GPT-4o generation artifacts without reading anomaly labels or test metrics; spring subsequences are normally constrained to 4–6 behavior quadruples to avoid GCAD-driven overlength. For strict prompt comparison, pass `--prompt-profile original`; when GCAD is disabled, that mode is byte-equivalent to upstream SmartGen. Every run records prompt hashes and a non-filtering environment-adherence summary in its manifest.

Codex runs with a read-only sandbox, explicit `none` reasoning effort, the current Codex login, and user/project CLI configuration disabled for experiment isolation. Upstream SmartGen requested `temperature=0`, `top_p=0`, `seed=2024`, and `max_tokens=8040`; the current Codex CLI rejects all four as unknown configuration fields. The run manifest therefore records those values as the upstream request and explicitly marks them as not applied instead of pretending they were fixed. The model slug, reasoning effort, prompt profile, experiment seed, exact prompt hashes, and outputs remain recorded and reproducible within the controls the Codex CLI actually exposes.

## Formal single-seed experiments

Accepted runs live under [`experiment_archive/formal/`](experiment_archive/formal/).
The registry and every archive manifest contain the exact configuration, metrics,
prompt/response provenance, GCAD state, TOF counts, and checksums.

| Dataset | Change | GCAD | F1 | Accuracy |
| --- | --- | --- | ---: | ---: |
| FR | winter → spring | require, ready | 0.961749 | 0.960227 |
| FR | daytime → night | auto, disabled | 0.971924 | 0.971113 |
| FR | single → multiple | require, ready | 0.994475 | 0.994444 |
| SP | winter → spring | require, ready | 0.971122 | 0.970467 |
| SP | daytime → night | auto, disabled | 0.984420 | 0.984174 |
| SP | single → multiple | require, ready | 0.913295 | 0.905063 |
| US | winter → spring | require, ready | 0.953375 | 0.952586 |
| US | daytime → night | require, ready | 0.889882 | 0.876256 |
| US | single → multiple | require, ready | 0.941746 | 0.938143 |

Historical calibration and failed prompt experiments are intentionally not kept in
the active tree. They remain recoverable from Git history when they were tracked.


