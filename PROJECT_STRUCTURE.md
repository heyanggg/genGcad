# SmartGen-GCAD project structure

## Active project

`SmartGen/main.py` is the only active SmartGen-GCAD experiment entry point. Its
pipeline is:

```text
source data
  -> TSS
  -> SSC/SPPC -----\
  -> original GSS --+-> Codex GPT-5.6 -> parse -> numeric conversion -> TOF
  -> GCAD ----------/                                      -> anomaly detection
```

GCAD reads the complete TSS source split before SSC. It contributes only stable
directional lag guidance and does not replace GSS or perform anomaly scoring.

The active implementation is organized as follows:

```text
SmartGen/
├── main.py                         experiment orchestration
├── experiment.py                   validated configuration and resumable run records
├── archiving.py                    immutable archive, integrity checks and promotion
├── codex_backend.py                isolated Codex GPT-5.6 invocation
├── gcad.py                         predictor, gradient graph and sparsification
├── split.py / dayse.py             TSS and day grouping
├── baseline2.py / sppc.py          SSC/SPPC
├── text_translation_matrix.py      original GSS
├── extract.py / transnumber.py     response parsing and numeric conversion
├── security_check.py               original TOF
└── baseline1.py                    anomaly detector
```

## Runtime directories

The original SmartGen functions still require their historical relative paths.
These directories are therefore working storage, not the canonical experiment
record:

| Directory | Purpose | Overwrite risk |
| --- | --- | --- |
| `SmartGen/IoT_data/` | source data, TSS/SSC/GSS intermediates, category outputs | source intermediates can be rebuilt/overwritten |
| `SmartGen/IoT_model/` | SSC model checkpoints | overwritten for the same source environment |
| `SmartGen/artifacts/gcad/` | cached GCAD extraction | fingerprint-protected but rebuildable |
| `SmartGen/runs/` | exact prompts, responses and generation manifest | isolated by run ID |
| `SmartGen/filter_data/` | merged numeric data and TOF outputs | isolated by artifact model/run ID |
| `SmartGen/check_model/` | TOF checkpoints | isolated by artifact model/run ID |
| `SmartGen/anomaly_runs/` | detector split, checkpoint and metrics | isolated by run ID and seed |

Do not use the presence of a runtime file as proof that it belongs to a particular
experiment. Use the experiment archive manifest and checksums instead.

## Experiment archive

`experiment_archive/` is the canonical human-facing record. Every archived run has:

```text
<status>/<archive-id>/
├── README.md
├── manifest.json
├── checksums.sha256
└── artifacts/
    ├── source_snapshot/
    ├── ssc/
    ├── gss/
    ├── gcad/
    ├── generation/
    ├── tof/
    └── anomaly_detection/
```

The metadata and checksums are tracked by Git. Heavy artifacts are local and ignored
by Git. `experiment_archive/registry.json` is the summary index.

Statuses have fixed meanings:

- `completed`: completed formal experiment.
- `verified_candidates`: high-scoring candidate; not yet a multi-run conclusion.
- `diagnostic`: debugging or prompt/backend calibration run.
- `failed`: failed run retained for diagnosis.

Complete generation+detection runs are automatically archived. A good completed run
can be promoted without copying its artifacts:

```bash
python SmartGen/archiving.py verify
python SmartGen/archiving.py promote <archive-id>
```

GCAD ablations must use an explicit prompt mode:

```bash
--gcad-mode auto       # use stable guidance when available
--gcad-mode off        # suppress GCAD guidance
--gcad-mode require    # fail if stable guidance is unavailable
```

## Upstream SmartGen material

The following top-level directories are retained from upstream SmartGen and are not
imported by `SmartGen/main.py`:

```text
ablation_study/
anomaly_detection_baseline/
anomaly_detection_pipeline/
behavior_prediciton_baseline/
behavior_prediciton_pipeline/
parameter_study/
```

They are reference baselines/studies, not SmartGen-GCAD run output.
