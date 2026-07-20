# Prospective replicate-2 source gate failure

The frozen request file was verified before generation at SHA256 `7c0fd1766598b4cfacdbcfac31fc1f672416dbc80c652ed4ddd8f4d9439c6860`. Tag `source-semantic-v1-frozen` points to commit `547156f`. Prompts, group counts, request counts, and source-semantic thresholds were not modified.

The Codex file workflow materialized 16 raw response records containing 137 newly authored sequences. Post-raw validation required zero repairs: JSON failures 0, schema failures 0, illegal devices/actions 0/0, missing responses 0, wrong lengths 0, exact duplicates 0, and 137 final valid sequences. Raw and validated SHA256 are identical: `561f24e708066ad0194cd48ab8cc1f649c7d1a22f2a0ea076d380e2cbb9ff964`.

Before raw materialization, ordinary authoring preflight corrected six exact template collisions and three complete copies of representatives from the sequence's own group. This preflight did not calculate source-semantic scores. No response was changed after the raw file was frozen.

## First generation gate

The original distribution gate failed only `source_exact_copies_zero`. All legality, uniqueness, length variation, source-relative entropy, template concentration, cross-group duplication, and maximum action-share checks passed. Generated length was 2/3/8 min/median/max; unique ratio 1.0; normalized bigram/trigram entropy 0.92404/0.98140; top-1/top-5 shares 0.00730/0.03650; maximum action share 0.15058.

Three short sequences independently matched representatives from a different SPPC group:

- `r2_0_1_04`: `Television:volumeDown → Television:setChannel`, matching source group `1_0`, index 26;
- `r2_1_1_04`: `Camera:notification → Blind:windowShade close`, matching source group `1_0`, index 14;
- `r2_4_1_14`: `Television:setInputSource → Television:setChannel`, matching source group `1_1`, index 0.

These are legal, unique generated records rather than repairable JSON, legality, count, or exact-duplicate failures. Changing them after seeing the gate would violate the prospective protocol. The experiment therefore terminates as **prospective replicate-2 source gate failure**.

`source_semantic_v1` was not run because the preceding gate had already failed. Consequently its five actual metrics are recorded as not run, not estimated or substituted. TOF, detector training, reconstruction loss collection, target behavior loading, and final evaluation were not run. No subset was selected and replicate-2 will not be regenerated.

Full local evidence is under `outputs/codex_generation_v2/fr/spring/baseline_source_semantic/replicate_2/`. Versioned evidence is under `experiment_artifacts/fr_spring_baseline_v2/replicate_2_source_gate_failure/`.

