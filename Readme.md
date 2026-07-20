# Readme

This document contains part of the source code for the paper *"SmartGen: Synthesizing Context-Aware User Behavior Data for Adaptive Smart Home Intelligence"*. It includes four functional modules (TSS, SSC, GSS, TOF), data synthesis procedures, testing on two smart home tasks, parameter experiments, and ablation studies. Please note the following:

1. In the code, **SPPC** refers to **SSC**. SPPC is an earlier naming version that was not changed due to the established workflow.
2. The **SmartGen** folder contains the implementations of the four functional modules as well as the data synthesis system. The other six folders correspond to various experimental setups.
3. Each folder includes a `main.py` file, which serves as the entry point for running the corresponding experiment. The CSV files in the `results` directory contain the recorded outcomes of these experiments.
4. This code does **not** provide APIs for large language models, but it **does** open-source various types of synthesized data and synthesis logs.

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

## Minimal source-only GCAD + Codex branch

The `minimal-gcad-codex` branch keeps official SmartGen intact except for the generation entry point in `SmartGen/main.py`. Two optional adapters are added under `SmartGen/extensions/`:

- `CodexFileBackend` exports the original SmartGen Prompt to text files and consumes Codex-authored text responses. It makes no external API call.
- `source_gcad` learns source-only directional scores; `gss_rerank` can reorder existing `action_transitions.json` edges. It never adds GCAD-only edges.

GCAD is off unless `--gcad-relation` is supplied. With GCAD off, the original GSS object and Prompt content are unchanged. TSS, SSC/SPPC, TOF, `baseline1.py`, `baseline2.py`, `security_check.py`, and the official anomaly detector remain unchanged.

From the repository root, learn a source-only relation file:

```bash
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m SmartGen.extensions.source_gcad \
  --source SmartGen/IoT_data/fr/winter/trn.pkl \
  --dataset fr \
  --output outputs/fr_winter_relation.json
```

Run SmartGen from `SmartGen/`. First export its Prompts:

```bash
python main.py --need_generate True --need_test False --model codex \
  --codex-mode export --codex-dir ../codex_io
```

Place Codex-authored responses in `codex_io/responses/day_*.txt`, then rerun with `--codex-mode consume`. For the GCAD arm, add:

```bash
--gcad-relation ../outputs/fr_winter_relation.json --gcad-alpha 0.2
```

No generated artifacts or experimental checkpoints are tracked on this clean branch.


