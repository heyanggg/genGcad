# FR winter→spring baseline protocol v2 results

## Experiment identities

- O: official filtered synthetic downstream sanity check; environment verification only.
- A1: SmartGen-compatible Codex-file baseline v1 smoke; not an official reproduction.
- A2: SmartGen-compatible grouped Codex-file baseline v2; not an official reproduction.

| Group | Precision | Recall | F1 | Accuracy | FPR | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| O, current environment | 0.7373 | 0.9886 | 0.8447 | 0.8182 | 0.3523 | 1.4874 |
| A1 smoke | 0.6350 | 0.9886 | 0.7733 | 0.7102 | 0.5682 | 0.00006175 |
| A2 grouped | 0.7647 | 0.4432 | 0.5612 | 0.6534 | 0.1364 | 4.2920 |

A1's FPR is reconstructed from its 88 normal/88 attack evaluation and saved precision/recall/accuracy: TN=38, FP=50, FN=1, TP=87. A2's saved matrix is TN=76, FP=12, FN=49, TP=39.

O confirms the current downstream chain can reproduce the official recorded F1 within 0.0167, so neither A1 nor A2's larger gaps should be blamed on a broken detector implementation. A1 collapses action templates and drives an extremely small threshold. A2 removes that collapse and greatly lowers false positives, but its validation threshold exceeds the median attack score and causes a major recall loss. A2 is 0.2122 F1 below A1 and is not a performance improvement.

The A2 generation and reconstruction gates passed before target evaluation, and the final target result was viewed once after hashes and settings were frozen. No target-informed rerun or threshold change was performed.

## GCAD readiness decision

Do not create `gcad-representation-v2` yet. Illegal GCAD semantic channels are now hard-rejected and ranking fairness is repaired, but A2 is not yet a sufficiently stable downstream reference: it replaces near-zero template collapse with a bimodal validation distribution and excessive threshold. A future baseline iteration must use a pre-registered, source-only semantic-coherence criterion and new generation replicate; it must not optimize against the A2 target result.

That criterion is now implemented as `source_semantic_v1`. Applied post hoc without opening target files, A2 fails all five source-semantic checks: only 36.79% event support, 3.08% transition support, 27.74% zero-anchor sequences, 41.61% group-anchor coverage, and 3.72× action-vocabulary expansion. A new request-only replicate is frozen, but no responses, TOF, or target evaluation have been run.

Compact evidence and hashes are in `experiment_artifacts/fr_spring_baseline_v2/`; full local score arrays and checkpoints remain under `outputs/codex_generation_v2/`.
