# SmartGen-GCAD — next research

This lightweight checkout is the `codex/next-research` workspace. Its starting
source was copied from the **working tree** at
`/home/heyang/projects/SmartGen_GCAD`; that tree had uncommitted changes.
`baseline_snapshot.json` records the copied files' original SHA-256 values.
Existing `context_research/` and ignored `tof_research/` files are separate work.

## Current experiment: RaSR-inspired augmentation only

The active protocol is `rasr_research/formal_experiment.py`: real source -> TSS ->
RaSR C/T/S insertion/replacement -> SSC -> existing Codex -> unchanged TOF and
anomaly detector. **There is no independent RaSR regenerator.** Codex remains
the only behavior generator. This borrows one part of RaSR; it is not a full
reproduction. Earlier CBRT/endpoint-transfer code below is historical and is
not used or enabled by this experiment.

`SmartGen/rasr.py` implements the directed co-occurrence counts, position-decayed
temporal counts (window 2, decay 0.8), cosine semantics and at most two semantic
plus two statistical edit attempts. Failed/no-op/duplicate edits are skipped.
Raw device/action MiniLM embeddings have no function/state-equivalence gating.
Times stay in original coarse slots and devices come from the action dictionary.

`SmartGen/ssc_input.py` accepts explicit I or I+J records and preserves provenance.
All arms use the same original SSC autoencoder checkpoint fitted only on true I.
The autoencoder is a compressor, not an extra behavior generator. Original GSS
and cached GCAD (source hash verified) read only true I; no augmented observations
are counted as real. Original data files are never overwritten.

The latest completed scenario is **one FR-DT B3 run (daytime -> night, seed 2024)**,
in `rasr_research/runs/formal-fr-dt-20261125/`. Its legacy archived hints confirm
disabled GCAD, zero relations and a fingerprint/source-TSS hash match; the old
config has no mode flag, which is not invented. Not an active-GCAD comparison.
Preserve the archived 7-call budget and 47 example upper bound (actual 47,
including 16 J naturally selected by original joint SSC at threshold 0.92).
Source pool: 319 I plus 392 valid unchanged-RaSR J. Original formal generation
has 28 rows, train/validation 22/6 and F1 0.9719244513; this is its full archived
scale, not the retired 8-9-row pilot. Keep original 10-epoch TOF, hour-field
15-epoch detector and p95. `formal_records/fr-dt-predeclared-experiment.json`
fixes the conditional generated-normal-only tuning search before new results.
No fresh B0, generation expansion/repeat, ablation or independent regenerator.
The very small validation set and historical disabled-GCAD state are explicit
limitations, not silently changed to improve apparent comparability.

FR-DT completed 7/7 calls, 28 generated rows, TOF first 27/final 28 and original
detector split 22/6 (historical raw/final/split also 28/28/22/6). Original B3
F1 **declined to 0.8964218456**, FP=220/FN=0, versus original historical
**0.9719244513**, FP=55/FN=0. This negative result is retained separately.
The predeclared 144-configuration generated-normal validation search selected
lr=0.001, wd=0, **5 epochs / mean error / p97.5**, shared across both pools.
Selection was locked and committed **before** tuned test confirmation:
`95d4ee464c8441e3bab99b8e903603de1f7e2aec`. No new Codex calls.
Tuned B3 F1 is **0.9916666667** (FP=16/FN=0, precision 0.9834710744,
recall 1, FPR 0.0168067227, FNR 0, AUROC 1, AUPRC approximately 1).
The same-tuned historical-data control is **0.9333333333** (FP=136/FN=0),
which is WORSE than its original protocol. B3 exceeds the original historical
reference by **1.974222 pp** and the common-tuned control by **5.833333 pp**.
Keep all four scores: recovery is not an unchanged-protocol RaSR gain or a
universal detector improvement. `FR-DT-B3-tuned-seed2024` is human-designated
formal; original B3 is supporting-negative. Only 6 held-out parent rows per
final model (69 correlated views), one final seed and exposed tests: exploratory.
GCAD stays verified disabled with no invented mode flag. Generation is closed.

The preceding completed scenario is **one SP-DT B3 run (daytime -> night, seed 2024)**,
in `rasr_research/runs/formal-sp-dt-20261125/`. Keep its archived GCAD-auto
decision: guidance was disabled, zero prompt edges. This is not an active-GCAD
three-module comparison. Do not silently enable GCAD to change the reference.
The fixed historical budget is 13 calls and 273 example slots. Original joint
SSC (threshold 0.917) naturally selects 195 J among 273 examples from 1327 I
and 3045 valid unchanged-RaSR variants. Keep original TOF, hour-field detector,
15 epochs and p95 calibration. Historical F1 is 0.9844200808, final count 51,
detector split 40/11. `formal_records/sp-dt-predeclared-experiment.json` freezes
the conditional normal-validation-only search before new results. No fresh B0,
ablation, independent generator, extra generation or test-label selection.

SP-DT completed 13/13 real calls: 52 generated rows, TOF first 48/final 52,
detector train/validation 41/11. Historical counts are 52/48/51 and 40/11;
one extra retained training row is disclosed, not treated as exactly matched.
With unchanged 15-epoch/p95 protocol, B3 F1 is **0.9912841371** versus
**0.9844200808** (**+0.686406 pp**), FP **54 -> 30**, FN **0 -> 0**.
Precision 0.9827188940, recall 1.0, FPR 0.0175849941, FNR 0,
AUROC=AUPRC=1.0 (ranking also 1.0 for the original checkpoint).
It is registered as `SP-DT-B3-original-seed2024`, explicitly retaining the
GCAD-auto-disabled caveat. Numeric p95 thresholds differ across separate fits
(original 0.0023697200, B3 0.0662797328). The gain is fewer false positives,
not improved ranking/recall or proof of stable causal RaSR benefit.
The conditional search was **not triggered**; zero extra generation or tuning.
All evidence is preserved and the generation policy is closed.

The preceding completed scenario was **one US-DT B3 run (daytime -> night, seed 2024)**,
in `rasr_research/runs/formal-us-dt-20261125/`. Historical GCAD is enabled with
11 prompt relationships. Its existing checkpoint reproduces F1 **0.9035871046**
exactly (AUROC 0.9417488117, AUPRC 0.9585462391), without B0 generation/training.
The fixed budget is **18 calls**, **450 actual examples / 450 upper bound**,
including **323 J** naturally selected by the original joint SSC (threshold
0.919). Source inputs: 1364 true daytime TSS rows plus 3825 valid unchanged-RaSR
variants. Original TOF, 15-epoch detector, hour-field input and **p93** calibration
are retained. Old seasonal inputs/queues cannot be reused for this scenario.
`formal_records/us-dt-predeclared-experiment.json` freezes the conditional
normal-validation-only tuning search before new results; no additional model,
generator, ablation, generation repeat or test-label-based selection is allowed.

The three previous large archives are now uploaded and verified by both exact
bytes and GitHub SHA-256 digests, with receipts committed in `formal_records/`:
[SP-ST / US-ST backup](https://github.com/heyanggg/genGcad/releases/tag/rasr-formal-sp-us-20261124)
and [FR-ST original + tuned backups](https://github.com/heyanggg/genGcad/releases/tag/rasr-formal-fr-st-20261124).
Raw source/target datasets and credentials are excluded. The 2026-11-25 run
starts only after all three prior asset checks pass.

US-DT is now completed and registered as **US-DT-B3-original-seed2024**:
18/18 successful real calls, 72 generated rows, TOF first 59/final 68 and
detector split 54/14. Historical raw/final/split counts are also 72/68/54/14.
At the unchanged original p93 protocol, B3 F1 is **0.9230055659** versus
**0.9035871046**, a **1.941846 percentage-point** gain. Precision is 1.0,
recall 0.8570198105, FPR 0, FNR 0.1429801895, AUROC=AUPRC=1.0.
FP drops 139 -> 0; FN stays 498. Numeric calibrated thresholds differ
(historical 0.0020601516, B3 0.2926191370); no target-derived cutoff is used.
Because the original-protocol result exceeds its reference, the predeclared
conditional tuning was **not triggered**: no tuning/refit or extra generation.
This remains one historical comparison, not evidence of stable causal gain.
The complete prompt/SSC/generated/TOF/detector evidence and ranking scores are
retained; generation policy is closed after exactly 18 new calls.
Posthoc read-only diagnostics show that hour code 3 occurs in B3 validation
but not detector training, and its row has the highest validation loss. With
only 14 validation rows, this affects p93 calibration. The gain is fewer false
positives, not better recall. See `formal_records/us-dt-posthoc-diagnostics.json`;
no test-score boundary is used to choose a new threshold.

The preceding FR-ST scenario was **one FR-ST B3 run (seed 2024)** versus its archived
Gen+GCAD F1 0.9617486339. It uses 16 fixed calls, original joint SSC threshold
0.918, 358 actual examples (209 J), unchanged 10-epoch TOF and 15-epoch detector,
and the original FR **p95.5**, not the SP/US p95. Source preparation retains
1728 real TSS rows and 2417 valid B3 variants with the unchanged RaSR algorithm.
No B1/B2 variants or control-arm matching are computed for this B3-only step.
The isolated output is `rasr_research/runs/formal-fr-st-20261124/`.
`formal_records/fr-st-contingency-before-results.json` fixes the source/generated-
validation-only tuning contingency before this B3 result, if original F1 does
not exceed the historical baseline. There is no extra generation or new model.

FR-ST is now completed: 16/16 real calls, 161 generated rows, TOF first 140/final
157, original detector train/validation 125/32. Original p95.5 B3 F1 was
0.7428571429 (FP=0/FN=36, AUROC=AUPRC=1.0) and is retained as supporting data.
The predeclared 144-configuration normal-validation-only search selected the
same lr=0.001, weight_decay=0.0001, 25-epoch/max-error/p97.5 setting as US.
After locking, FR B3 F1 reached **0.9887640449** (TP=88/TN=86/FP=2/FN=0,
precision=0.9777777778, recall=1.0, FPR=0.0227272727, FNR=0,
AUROC=1.0, AUPRC=0.9999999999999998). The same-tuned historical-data B0 has
F1 **0.9617486339** (FP=7/FN=0), identical F1 to the original reference.
Thus the registered tuned FR group exceeds both references by **2.701541 pp**.
This is one seed on 88 normal/88 attacked tests, previously exposed; it is not
a fresh confirmatory trial or an isolated stable causal RaSR-gain claim.
All normal split changes (tuned B3 126/31), correlated views, candidate metrics,
original negative scores and weights are preserved. No extra Codex calls followed.

The preceding completed scenario was **one US-ST B3 run (seed 2024)** versus the
original formal US Gen+GCAD archive. It has **11** fixed generation tasks, the
unmodified joint SSC's **159** actual examples (95 J), and a historical example
upper budget of 229. No refill, SSC tuning, fresh B0, ablation, repeated run or
additional SP generation is authorized. The active generation policy is
`rasr_research/generation-policy.json`; it applies to child queue roots too.
After the user's explicit next-scenario approval, the hold was lifted for this
US queue only, then re-closed after all 11 fixed tasks had valid responses.
Both completed queues are blocked from additional generation.
Previous quota-failed logs are retained in each
call's `prior-failed-attempt-before-explicit-credit-resume` directory.
No automatic retry is enabled. The native
worker stops the queue on its first failure and preserves all failed-call logs.

The completed SP-ST B3 queue had the historical 16-call budget, 399 actual Prompt
examples selected by original joint I+J SSC, including 233 J examples naturally
selected (not a forced quota). Historical SP F1 is 0.9711215581; its 205 TOF-final
sequences split into 164 detector-training and 41 validation rows. Same original
prompt text outside examples, SSC checkpoint/threshold, source GSS/GCAD, 10-epoch
TOF, 15-epoch detector and generated-validation p95 calibration are retained.
Generation output count is observed, not forced to equal the historical 216 raw
or 205 post-TOF rows. A single historical comparison cannot establish stability.

These commands are **offline preparation/evaluation only**, with zero Codex calls:

```bash
python rasr_research/formal_experiment.py --action prepare-single --dataset sp
python rasr_research/formal_experiment.py --action evaluate-single --dataset sp
python rasr_research/formal_experiment.py --action score-historical --dataset sp
```

For the authorized US step, use the same actions with `--dataset us`.
`wait-single --dataset us` only waits for the 11 already-queued responses and
then executes original TOF/detection; it never submits generation requests.
US reports use `active-scope-us.json` / `single-b3-comparison-us.json`; existing
SP result and report filenames are preserved. Historical US F1 is 0.9533746332;
re-scoring its original saved detector at the original threshold reproduced all
recorded metrics exactly (AUROC 0.9692301426, AUPRC 0.9494183299), with zero new
generation or training. The 159-vs-229 SSC example difference is part of the
pipeline comparison and must be disclosed, not interpreted as an isolated
algorithmic treatment effect.

The completed US B3 run **did not improve**: F1 0.8728916710 vs the specified
historical 0.9533746332 (-8.0483 percentage points). FP increased 194 -> 225;
FN increased 92 -> 506. Precision/recall were 0.9177330896 / 0.8322281167;
FPR/FNR were 0.0746021220 / 0.1677718833; AUROC/AUPRC were 0.9499873904 /
0.9115929604. The unchanged TOF retained 89 of 95 generated sequences; detector
training/validation used 71/18. These original-protocol negative results, raw
outputs and checkpoints are retained unchanged. Later, separately recorded
validation-only detector tuning is documented below; it does not replace them.

An operational Windows GBK stderr-decoding failure affected the first three
US tasks. Their complete real stdout was recovered and validated (32 sequences),
with **zero repeated model requests**. Original CLI exit codes were unavailable
and explicitly recorded as unknown, not invented as zero; original error files
and stdout SHA-256 fingerprints remain in the audit. The remaining eight tasks
used explicit UTF-8 pipe decoding and normal zero-exit receipts. This only fixes
I/O; model, prompt text, RaSR algorithms and downstream parameters did not change.
The SP result/report SHA-256 fingerprints were re-checked unchanged. Across these
single historical comparisons, current evidence does not establish stable RaSR
gains; US decreases are not discarded or hidden by SP's historical improvement.

`active-scope.json` and `single-b3-comparison.json` in the formal output folder
record the archived baseline fingerprint, protocol checks, actual J exposure,
current blockers and (when available) the real B3 metrics and differences.
`score-historical` only loads the already-trained original formal detector and
its original fixed threshold to verify recorded metrics and save ranking scores;
it neither trains nor generates, and never overwrites the archived baseline.
The old 18-run CLI queue/watch actions are disabled. Previously produced results
are retained but are not substituted for the requested original formal baseline.

The completed SP-ST B3 run generated 191 sequences; original TOF first kept 170,
then restored candidates to a final 184 (147 detector-training / 37 validation).
F1 was **0.9820193638**, compared with the specified original archive's
**0.9711215581** (+1.0898 percentage points). FP decreased 38 -> 8, but FN increased
5 -> 18: this is not an across-the-board improvement. B3 precision/recall were
0.9888579387 / 0.9752747253; AUROC/AUPRC were 0.9959545949 / 0.9949128022.
The original archived detector was re-scored without retraining: all previously
recorded metrics matched exactly, and its AUROC/AUPRC were 0.9826787224 /
0.9791831466. Target file hashes and the p95 validation protocol were checked.
No parameters were selected using these test results and no extra runs followed.

For transparency, an already-completed fresh B0 from the superseded schedule
(not rerun here) had F1 0.9945355191 and 244 TOF-final sequences. It is retained
alongside the requested historical comparison, not concealed or substituted.
Thus this one B3 exceeds the specified historical baseline, but **does not
establish stable or causal RaSR gains**, nor does it beat every existing B0 run.

### US detector diagnosis and offline tuning (2026-11-24)

This is a separate detector protocol, **not a replacement for the
unchanged original-protocol RaSR results above**. No new Codex calls, no new network/loss,
no changes to RaSR, SSC, GSS, GCAD or TOF. Existing generated normal corpora and
the existing TransformerAutoencoder are reused for both historical B0 and B3.

`rasr_research/diagnose_formal.py` found five validation action types (seven
events) absent from the original B3 training partition; these occur in its
highest-loss normal validation rows. This undermines the p95 calibration.
Source TSS is also much shorter than the generated 5-6-action chains:
23625/30105 source sequences contain only one or two actions. These are
diagnostic observations, not proof that any single change caused the recovery.

`rasr_research/tune_detector.py` splits exact-duplicate normal groups together,
retains training support for validation actions, and searches 144 existing-model
settings on three normal holdouts in **both** arms. Generic unseen-legal-action
corruptions serve only as validation proxies, not true attack labels. A second
validation family includes held-out normal prefixes/singletons for calibration;
the parent rows, not the correlated views, are the independent units. Its choice
was recorded in `family-choice-before-test.json` before final test scoring.

The shared locked setting is Adam lr=0.001, weight_decay=0.0001, 25 epochs,
maximum per-action reconstruction error, and a 97.5th-percentile normal-view
threshold. B3 F1 changes 0.8728916710 -> **0.9597454256**, recall
0.8322281167 -> 1.0, but FPR worsens 0.0746021220 -> 0.0838859416
(FP 225 -> 253; FN 506 -> 0). Common-tuned historical-data B0 reaches
**0.9592875318**; B3 exceeds it by only **0.045789 percentage points**.
This does not establish a stable RaSR-specific gain. Target tests were already
exposed in earlier experiments: the locked final scoring is exploratory, not a
fresh confirmatory holdout. No second candidate was chosen by target test F1.

All candidates, training curves, normal splits, selected model weights, locked
parameters, continuous final scores and seven metrics are retained under
`rasr_research/runs/tuning-20261124/`. The original formal records are untouched.
There are three validation seeds but only **one** final evaluation seed.

```bash
# Read/aggregate existing results only; no training, generation or test selection.
python rasr_research/tune_detector.py --dataset us --action report \
  --output rasr_research/runs/tuning-20261124/covered-view-cv
```

`fit` refuses to overwrite a locked selection; `confirm` refuses to overwrite an
existing evaluation. Do not apply this new detector protocol silently to the
formal pipeline or claim that a detector-tuning improvement is a RaSR gain.

### Human-designated formal experiment registry

The user's clarified formal groups are **SP-ST B3 F1 0.9820193638** and
**US-ST tuned B3 F1 0.9597454256**, not two US-ST variants. Their historical
Gen+GCAD references are 0.9711215581 and 0.9533746332 respectively. The registry
and portable result copies are in `rasr_research/formal_records/registry.json`.
Formal archival status does not convert the tuned US result into an unchanged-
protocol comparison or a fresh confirmatory test. Its same-tuned-protocol B0
F1 0.9592875318 is disclosed alongside the requested historical comparison.
The lower US B3 result, unselected tuning family and every existing candidate
remain supporting records; they are not deleted, relabeled as successes or
overwritten. Future above-baseline B3 results must retain their exact protocol,
all attempts, weights, provenance and limitations when added to this registry.

The next completed group, **FR-ST tuned B3 F1 0.9887640449**, was added under that
rule. Its original-protocol F1 0.7428571429 is a separate supporting record.

`rasr_research/save_formal.py` creates a SHA-256/ZIP-CRC-verified portable backup
of generated data, weights, prompts/responses, all tuning candidates and results.
Raw source/target datasets, credentials and unrelated research are excluded.
Compact results and code belong on `codex/next-research`; large artifacts are
uploaded as a GitHub Release asset, with a checked receipt in `formal_records`.

### Historical pilot and retired multi-arm schedule

B0 = I, B1 = count-matched random edits, B2 = semantic edits, B3 = semantic plus
statistical edits. B1-B3 match retained augmentation counts **per source parent**.
B4/B5 are explicitly disabled by the user's latest instruction. A 90/10 original
parent-group split precedes graph fitting; validation parents produce no J.
The first pilot ran B0/B3 on SP-ST and US-ST. Each arm had two Codex calls;
Both datasets use four examples per call. Augmented arms allocate half of the
same total prompt budget to J. This stratification and frozen SSC eval mode are
explicit controls, not claims about original RaSR. Actual IDs/text/J counts are
saved in `prompt_exposure.json`, alongside original responses and detector scores.

```bash
# Historical commands only. Do not run under the current single-B3 scope.
python rasr_research/run_experiment.py --dataset sp \
  --output rasr_research/runs/pilot-20261124 --examples 4
python rasr_research/run_experiment.py --dataset us \
  --output rasr_research/runs/pilot-20261124 --examples 4
```

The following native bridge command was used for the historical pilot; it is
not the current scope (and is blocked by the active generation policy):

```powershell
python rasr_research/codex_worker.py --root rasr_research/runs/pilot-20261124 `
  --codex-bin C:/Users/heyang/AppData/Local/OpenAI/Codex/bin/23f7d7f110b19ac3/codex.exe
```

Model/embedding/short-sequence adaptations are saved in each prepared manifest.
Resume reuses completed responses, never silently repeats a failed CLI call or
substitutes another model. Small single-seed pilots cannot establish significance.

## Earlier relation audit (historical; not the active RaSR path)

- Optional **additive** relation audit in `SmartGen/main.py`. Original GSS,
  selected GCAD edges, GCAD wording, SSC examples and environment prompt are
  preserved. Removing the extra block restores the baseline prompt byte for byte.
- Frozen, inspectable device/command labels, explicit abstention for unresolved
  functions/states, separate GSS/GCAD evidence and source-only time histograms.
- Source fingerprints, immutable run-scoped relation artifacts, archive inclusion,
  support checks, per-target/total limits, and isolated filenames per mode.
- Pinned MiniLM action embeddings, compatibility masks, sparse endpoint transport,
  and a nine-task feasibility audit. **Transport candidates are currently an
  offline research artifact, not injected into generation.** See the feasibility
  decision in `relation_research/IMPLEMENTATION.md`.

No claim of detection improvement follows from these implementation checks.

## Runtime

Use the existing WSL environment; no environment installation was required:

```bash
cd /mnt/c/Users/heyang/Documents/smartgen-gcad
/home/heyang/miniconda3/envs/smartguard_env/bin/python -m pytest -q
```

Source data, checkpoints and archived experiments were not copied into this
lightweight checkout. Source-only previews read the original project and write
only into this workspace:

```bash
python relation_research/run_previews.py \
  --source-project /home/heyang/projects/SmartGen_GCAD \
  --output relation_research/previews/phase1_final
```

This checks both `typed` and `adapted` on all nine tasks with formal GCAD source
artifacts. It does not train, generate, or inspect target data.

## Embeddings and transfer feasibility

The pinned encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with attention-masked mean pooling
and L2 normalization. This follows the encoder's model card. It is an engineering
substitute for semantic retrieval, not a reproduction of RaSR's LLM encoder.

Windows can download the approximately 90 MB public weights (including LFS
SHA-256 verification) when WSL cannot reach Hugging Face:

```powershell
.\relation_research\download_model.ps1
```

Then, in WSL:

```bash
python relation_research/build_embeddings.py \
  --model-dir relation_research/models/minilm-1110a243 \
  --output relation_research/embeddings/minilm-v1

python relation_research/audit_transfer.py \
  --source-project /home/heyang/projects/SmartGen_GCAD \
  --embedding-dir relation_research/embeddings/minilm-v1 \
  --output relation_research/previews/cbrt_final
```

`build_embeddings.py --download` supports download directly from WSL when that
network is available. Inference always uses local files, safetensors and
`trust_remote_code=False`. Model weights, embeddings and large preview artifacts
are ignored by Git; source scripts and the compact feasibility result are kept.

## Generation integration

After staging the required **source training** files into this checkout's
`SmartGen/IoT_data/<dataset>/<source>/` and creating `SmartGen/IoT_model/`, the
existing runner supports:

```bash
python SmartGen/main.py --need-generate true --need-test false \
  --dataset fr --ori-env winter --new-env spring \
  --run-id relation-audit-1 --relation-mode adapted
```

Modes:

| Mode | Extra prompt content |
| --- | --- |
| `off` (default) | None; baseline prompt and filename behavior |
| `typed` | Source evidence organized by type, plus frozen action labels |
| `adapted` | The audit subset plus tentative target-environment policies |

`--relation-min-support`, `--relation-max-relationships`, and
`--relation-max-per-target` control **the extra block only**. The original GSS and
GCAD selection are not trimmed by these options. Unresolved labels are excluded
from the `adapted` supplement, but remain present wherever the baseline includes
them. An empty supplement leaves the baseline prompt unchanged.

These modes support controlled diagnostics. They are not the full CBRT proposal.
No `transfer` generation mode is exposed pending a stronger feasibility result.
