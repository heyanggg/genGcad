# Readme

This document contains part of the source code for the paper *"SmartGen: Synthesizing Context-Aware User Behavior Data for Adaptive Smart Home Intelligence"*. It includes four functional modules (TSS, SSC, GSS, TOF), data synthesis procedures, testing on two smart home tasks, parameter experiments, and ablation studies. Please note the following:

1. In the code, **SPPC** refers to **SSC**. SPPC is an earlier naming version that was not changed due to the established workflow.
2. The **SmartGen** folder contains the implementations of the four functional modules as well as the data synthesis system. The other six folders correspond to various experimental setups.
3. Each folder includes a `main.py` file, which serves as the entry point for running the corresponding experiment. The CSV files in the `results` directory contain the recorded outcomes of these experiments.
4. The original code did not include a usable large-language-model connection. This branch invokes GPT-5.6 through the locally authenticated Codex CLI.

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

The predictor must outperform a held-out frequency baseline. If the data are too small, validation fails, or no edge is stable, GCAD writes a disabled artifact with an empty relationship list; SmartGen then continues with the original GSS guidance. A successful relationship records its raw and normalized strength, lag, support, and seed stability. GCAD anomaly scoring is intentionally not included.

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
  --experiment-seed 2024
```

The artifact label for this example is `gpt-5.6-sol__run1`. GCAD still uses its three internal stability seeds, while `--experiment-seed` controls SSC, TOF, the GCAD holdout split, and anomaly-detector training. The anomaly detector writes its split, checkpoint and `metrics.json` under `SmartGen/anomaly_runs/` instead of overwriting target-domain source data.

By default, GCAD writes `SmartGen/artifacts/gcad/<dataset>/<original_environment>/gcad_hints.json`. The artifact includes a source-data hash and extraction configuration, so an unchanged result is reused instead of retrained. Pass `--gcad-force` to rebuild it. The original `action_transitions.json` is only read and is never reranked or overwritten by GCAD. Only the compact relationship list—not training diagnostics—is sent to the model, and it is labeled as soft predictive guidance. Codex runs with a read-only sandbox, explicit `medium` reasoning effort, and the current Codex login; no prompt/response exchange directory or OpenAI Python SDK is required.

## Verified experiment

The legacy `FR / winter → spring / SPPC / threshold 0.918 / seed 2024` generation run completed end to end before run IDs were introduced. GCAD retained five relationships stable across all three internal seeds, all 16 Codex generation groups parsed successfully, and TOF retained 204 valid sequences.

The original SmartGen anomaly detector was then trained with 163 generated sequences and validated with 41. On 88 normal and 88 attack samples it produced `TP=87`, `TN=88`, `FP=0`, and `FN=1` (`accuracy=0.9943`, `F1=0.9943`). The machine-readable metrics are in [`SmartGen/anomaly_runs/fr_spring_gpt-5.6-sol_seed2024/metrics.json`](SmartGen/anomaly_runs/fr_spring_gpt-5.6-sol_seed2024/metrics.json).

The `FR / daytime → night / SPPC / threshold 0.92 / seed 2024` run also completed end to end. Only one of three GCAD predictor seeds passed the held-out quality gate, so the stability filter correctly disabled GCAD guidance and generation continued with the original GSS fallback. All seven Codex groups parsed successfully, producing 49 numeric sequences, of which TOF retained 48. The anomaly detector used 38 training and 10 validation sequences; on 952 normal and 952 attack samples it produced `TP=760`, `TN=899`, `FP=53`, and `FN=192` (`accuracy=0.8713`, `F1=0.8612`). The machine-readable metrics are in [`SmartGen/anomaly_runs/fr_night_gpt-5.6-sol__seed2024_SPPC_th-0.92_p-95_seed2024/metrics.json`](SmartGen/anomaly_runs/fr_night_gpt-5.6-sol__seed2024_SPPC_th-0.92_p-95_seed2024/metrics.json).


