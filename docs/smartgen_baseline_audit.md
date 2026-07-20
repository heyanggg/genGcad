# SmartGen baseline audit

Audit date: 2026-07-20. Repository root: `/home/heyang/projects/SmartGen_GCAD`; official upstream: `horizonsinzqs/SmartGen`; baseline commit: `c2ed36c` on `main`.

## Executable path

- TSS is `SmartGen/split.py::Split` (lines 115–129), backed by `split` (68–112). Input `IoT_data/<dataset>/<source>/trn.pkl` is a pickle of flat integer lists. Every four integers encode `[day, 3-hour-bin, device, action]`; TSS reshapes to four rows (72–74), splits on interval/total duration, then writes `split_trn.pkl` (118–122). Example: `[0,0,13,76,0,1,13,77]` is two events.
- SSC is named SPPC in code, as the official `Readme.md` states. `SmartGen/sppc.py::SPPC_select` (91–135) loads day files and `IoT_model/Transformer_<dataset>_<source>_15epoch.pth`, compares flattened encoder memories by cosine similarity, and writes `trn_day_<d>_SPPC_th=<threshold>.pkl`. The first-cell checkout contains precomputed day outputs but not the referenced `SmartGen/IoT_model` directory/checkpoint.
- GSS is `SmartGen/text_translation_matrix.py::ATM` (112–127). It reads the **complete** TSS output, takes action positions 3,7,… (116–121), and `LinkAnalyzer.fit_sequences` counts adjacent actions (20–36). `analyze_link` writes an object keyed by action; each value has a message and up to five `{next_action,count}` records (83–107).
- The real orchestration is `SmartGen/main.py` (87–162): TSS → day split/SSC → GSS/text conversion → prompt/API → parse/numeric conversion → TOF → downstream evaluation. `LLM_call` is at 49–80; prompt assembly is 136–151; raw response is pickled at 153–154 and parsed by `Extract`/`Transnum` at 156–157. The original API placeholders are retained but are not used by this integration.
- Original TOF is `SmartGen/security_check.py::security_check` (330–376). Stage 1 trains for 10 epochs on generated sequences and applies reconstruction-loss IQR (`detect_outliers_iqr`, 77–89; `check_outlier`, 286–328). Stage 2 splits retained generated sequences, retrains, reinserts each candidate only when generated-validation loss does not worsen. `security_check_file` (378 onward) is an explicit-path, CPU-compatible wrapper preserving that logic.
- Downstream AD is `anomaly_detection_pipeline/Anomaly_Detection_pipeline_model.py::Anomaly_detection` (283 onward): a Transformer autoencoder trains 15 epochs (`train`, 116–156); `find_threshold` (158–189) takes the configured percentile of **synthetic validation** losses; `evaluate` (191–281) reads target normal and labeled attack only for final metrics. The wrapper in `gcad_source/downstream_evaluation.py` makes this timing enforceable.

## Cells, files, and reference settings

FR, SP and US each contain `winter`, `spring`, `daytime`, `night`, `single`, and `multiple` directories under `SmartGen/IoT_data`. The mappings are winter→spring, daytime→night, and single→multiple. Official synthetic files are under `anomaly_detection_pipeline/synthetic_data`; downstream checkpoints are under `anomaly_detection_pipeline/check_model`. The official README records the nine compression/AD percentiles; FR winter→spring is 0.918 and 95.5. Historical FR spring synthetic count was audited as 137 before TOF and 125 after filtering, so the new experiment generated 137 per prompt arm. Old synthetic contents were neither used nor copied.

FR winter→spring was selected because source/TSS/day/GSS files and target final-evaluation files are complete and its scale is modest. The seven existing representative files contain 37,46,40,39,69,38,36 sequences (305 total). The integration archives one combined prompt instead of replaying the official per-day API loop, because the official SPPC checkpoint and an exact clean regeneration path are missing; this deviation is explicit.

## Reproduction probes

```bash
/home/heyang/miniconda3/bin/conda run -n smartguard_env python -m SmartGen.gcad_source.cli tensorize --input SmartGen/IoT_data/fr/winter/split_trn.pkl --output outputs/gcad_source/fr/spring/tensor --dataset fr --config configs/gcad_source/fr.yaml
/home/heyang/miniconda3/bin/conda run -n smartguard_env python -m SmartGen.gcad_source.cli continue-pipeline --directory outputs/codex_generation/fr/spring/baseline/replicate_1 --dataset fr --context spring --tof-epochs 10 --stable-relation outputs/gcad_source/fr/spring/relations/stable/stable_relation.json
```

Folders outside `SmartGen/` are experiment variants/historical copies; the audited primary path above is authoritative for compatibility.
