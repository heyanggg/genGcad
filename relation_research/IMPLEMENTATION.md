# Relation-guided research implementation and decision

Date: 2026-11-24 (Asia/Hong_Kong).

## Revision following the second external review

The original GSS and GCAD prompt blocks are preserved exactly. The initial
replacement approach has been removed. `typed`/`adapted` are now optional,
additive diagnostic modes, with `off` remaining the default. They should not be
presented as an independently validated paper contribution.

The new research operation is sparse semantic endpoint transport, evaluated
offline before any generation experiment. RaSR supplies the semantic-relation
idea; source relation extraction remains GSS/GCAD-derived. No learned
regenerator, source augmentation or target-data fitting is included.

## Reproducible semantic inputs

Complete action descriptions include device, command, function, desired state
and the original action string. Function/state fields come from versioned
lexical annotations, not measured target preferences or independently reviewed
device manuals. Unknown parameter values and unspecified controlled appliances
remain unresolved. Approximately half the dictionary actions remain unresolved;
this restricts the conclusions of the feasibility audit.

A frozen MiniLM encoder produces 384-dimensional vectors with attention-mask
mean pooling and L2 normalization. Its pinned public revision and all downloaded
file hashes are recorded in each embedding artifact. Text, dictionary, semantic
label, code and vector fingerprints are recorded. The original RaSR LLM was not
replicated.

The compatibility mask requires matching function, operation and desired state
across different catalogue device types. It rejects opposite states, ambiguous
toggles, notification/refresh operations, same-device aliases and unresolved
functions. A compatible catalogue entry is not proof that the device is installed
or preferred in the target home.

Cosine threshold: 0.65; at most three nonidentity neighbors per action. These are
research defaults, selected without target data or detection results.

## Candidate transport

Each mapping row contains identity weight 1 and positive cosine weights for the
allowed nonidentity neighbors, normalized by the row sum. The resulting sparse
matrix P is a transport coefficient matrix, not a calibrated probability model.

For every reliable source edge, the implementation enumerates endpoint choices
from P and accumulates contributions separately for GSS and GCAD. This implements
the sparse contributions to `P^T C P` and `P^T G^(lag) P`. By default, only one
endpoint changes. Two-endpoint transport is available in the offline audit as
`--max-changed-endpoints 2`. Identity-only results and collapsed self loops are
excluded.

Source weights:

- GSS: empirical next-action conditional frequency, after minimum count 3.
- GCAD: raw gradient strength divided by the maximum eligible GCAD strength,
  multiplied by the upstream fraction of stable seeds; minimum support 3 and the
  upstream seed-stability gate still apply.

Each contribution is the source weight times both endpoint transport weights.
The environment factor is currently only a catalogue legality gate (1 for
accepted mappings). **There is no learned or independently supported target
environment suitability score.** Rankings remain separate by relation type.
Accumulated scores are not probabilities and can exceed 1.

Candidate records retain original evidence, parent action names, mapping weights,
source-lag proposal, lag unit, source match count and an explicit
`target_observation_count: null`. A candidate with a valid existing source match
is not counted as novel and is not selected as new relation information. No
candidate is currently injected into generation or added to training data.

## Feasibility result and decision

See `feasibility_summary.json` and the ignored full artifacts under
`previews/cbrt_final/`.

| Dataset | Dictionary actions | Compatible directed mappings | Functional family |
| --- | ---: | ---: | --- |
| FR | 222 | 6 | Television ↔ Projector |
| SP | 234 | 6 | Television ↔ Projector |
| US | 268 | 6 | Television ↔ Projector |

The six directed mappings are three bidirectional pairs: switch on, switch off,
and audio mute. All six pass the 0.65 cosine threshold; the compatibility mask
already limits the pool to these six. Thus this audit does not establish an
incremental benefit of embeddings over the compatibility rules. Counts below
refer to source graphs after legality and minimum-support checks:

| Task | GSS edges with mapping / eligible | GCAD edges with mapping / eligible | Novel candidate edges, both types |
| --- | ---: | ---: | ---: |
| FR winter → spring | 3 / 51 | 0 / 5 | 3 |
| FR daytime → night | 0 / 9 | 0 / 0 | 0 |
| FR single → multiple | 4 / 22 | 0 / 1 | 4 |
| SP winter → spring | 6 / 86 | 0 / 2 | 7 |
| SP daytime → night | 3 / 33 | 0 / 0 | 3 |
| SP single → multiple | 5 / 55 | 0 / 1 | 6 |
| US winter → spring | 53 / 223 | 8 / 44 | 58 |
| US daytime → night | 22 / 106 | 0 / 19 | 29 |
| US single → multiple | 25 / 90 | 7 / 16 | 40 |

Multiple source edges can expand through the same mapping, so 58 candidate
relations does not mean 58 independently discovered mappings. The candidate
counts are not validated target relations, semantic-correctness measurements, or
performance gains. No target-specific mapping is established for season,
day/night, or occupancy changes.

Decision: **hold broad CBRT generation experiments and retain transport as an
offline feasibility tool.** The present evidence is too narrow to support a new
method claimed across all nine tasks. This does not prove that richer,
independently reviewed capability information could not produce other valid
mappings. The strict current annotation/mask is a material limitation. A narrowly
scoped US entertainment-device study remains a possible future experiment, but
would answer a smaller question and has not been launched.

## Data and indexing boundaries

- Only source `split_trn.pkl` and its matching formal GCAD graph are read.
  No target training, validation, test, attack or detection-label data are read.
- Preview GSS uses the same pure function as `ATM()`, including its first-ten-
  events limit and top-five successors per source action. Denominators include
  all outgoing raw transitions, not only the displayed top five.
- GSS lag 1 counts raw event positions. GCAD lags count valid device behavior
  positions after non-device/invalid device-action entries are excluded. The
  two meanings stay explicit even when evidence shares an action pair.
- Three-hour bins describe source context. They do not recover exact timestamps,
  multiweek durations, or target-environment invariance.
- Device/action mismatches are audited and excluded from **new** evidence;
  original source records and baseline prompts remain unchanged. US has some
  Heater device-ID inconsistencies. No automatic dictionary repair is performed.

## Validation and remaining evidence

Tests cover compatibility, immutable embedding fingerprints, transport factors,
source-vs-target evidence, source relation ablations, lag indexing, observed
candidate exclusion, support/stability gates, resume checks, generation-entry
integration with a mocked backend, and exact additive prompt preservation.

All 88 original and new unit/integration checks passed in the existing WSL environment.
Real-data previews cover nine tasks in two modes; the real-embedding feasibility
audit covers all nine tasks. Source training hashes remain unchanged.

No actual LLM generation, TOF retraining, downstream retraining or F1 comparison
has been performed. A generation-entry test double verifies wiring, not generation
quality. GCAD's independent downstream gain remains to be established by a new
controlled ablation; the retained project summary does not claim it is proved.

References: [RaSR paper](https://ojs.aaai.org/index.php/AAAI/article/download/38540/42502),
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
