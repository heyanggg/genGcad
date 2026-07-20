# Source-only semantic coherence gate

## Why this gate exists

The original A2 gate successfully rejected illegal actions, exact copies, duplicates, fixed lengths, and template concentration. It did not measure whether a legal generated combination remained connected to observed source behavior. A2 consequently used 145 actions and 290 transitions even though the cleaned FR winter TSS contains only 39 observed actions and 94 observed transitions.

This new gate is adopted after the completed A2 experiment. Although its inputs and thresholds are source-only, it must not be described as an A2 pre-target gate or used to revise the already viewed A2 target result. It is frozen for future generation replicates.

## Inputs and calibration

The command accepts only an explicit generated directory, dataset name, and complete source TSS path. It verifies that every request declares `uses_target_behavior=false` and that the complete source file and all SPPC group files come from the same source-context directory. No target-normal, attack, label, official synthetic, or A2 score file is opened.

Invalid semantic events such as `None:location` are removed before calibration. FR winter has 1,728 raw sequences, 923 usable sequences, 39 actions, and 94 adjacent transitions after filtering.

Calibration deterministically holds out one source day at a time and measures coverage by the other six days. Mean cross-day action coverage is 0.97770, mean transition coverage is 0.71397, and the zero-anchor share is 0.01733. The frozen policy uses deliberately conservative context-change tolerances:

- minimum generated action coverage: 50% of cross-day source coverage = 0.48885;
- minimum generated transition coverage: 25% of cross-day source coverage = 0.17849;
- maximum sequences without any source action: `min(0.25, max(0.10, 5 × source rate))` = 0.10;
- at least 50% of generated sequences must contain an action from their own SPPC group;
- generated action vocabulary may be at most twice the source vocabulary.

These formulas are fixed in `source_semantic_v1`; none uses a target anomaly score, target threshold, F1, or official synthetic distribution.

## A2 post-hoc result

A2 fails all five checks: action coverage 0.36789, transition coverage 0.03085, zero-anchor share 0.27737 (38/137), group-anchor share 0.41606, and action-vocabulary expansion 3.71795. The weakest groups are `1_0` and `6_1`, where only 1/6 of generated sequences have a group source anchor.

This explains a limitation missed by the first gate: A2 is structurally diverse but many sequences are effectively unconstrained combinations of static legal actions rather than group-conditioned behavior adaptations.

## Enforcement and next requests

`continue-pipeline` now refuses v2 TOF unless both `generation_quality_gate.json` and `source_semantic_gate.json` exist, pass, and declare zero target use. The grouped exporter records a semantic envelope in every request and Prompt:

- at least one group-source action per generated sequence;
- at least half of batch events from the group-source action vocabulary;
- group-observed adjacent transitions where applicable;
- no complete source-representative copy;
- bounded novel action types.

A new replicate-2 request set has been exported locally with 15 groups, 16 requests, and 137 requested sequences. Its request SHA256 is `7c0fd1766598b4cfacdbcfac31fc1f672416dbc80c652ed4ddd8f4d9439c6860`. Responses have not been generated, TOF has not run, and target behavior has not been evaluated.

```bash
python -m SmartGen.gcad_source.cli gate-source-semantics \
  --directory OUT --dataset fr \
  --source SmartGen/IoT_data/fr/winter/split_trn.pkl
```

