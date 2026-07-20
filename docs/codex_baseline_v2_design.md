# SmartGen-compatible Codex-file baseline v2 design

## Identity and boundary

A2 is a **SmartGen-compatible Codex-file baseline**, not an official SmartGen/GPT-4o reproduction. It restores the official organization of the available SPPC intermediate artifacts while using manually authored, offline Codex file responses. It does not call an external API and does not use official synthetic contents, A1 responses, GCAD-GSS data, target normal behavior, or target attacks for generation.

Static target device/action legality metadata and the winter→spring context description remain allowed. Target behavior is first opened by `evaluate-prepared`, after the response set, TOF output, detector, threshold, both quality gates, configuration, and hashes are frozen.

## Grouped request protocol

The official precomputed FR/0.918 SPPC artifacts yield 15 source groups with representative counts `30,7,30,16,30,10,30,9,30,30,9,30,8,30,6`. A2 exports one group-local prompt for each group. Group `4_1` has a 29-sequence allocation and is serialized as two independent response requests of 15 and 14, so the persisted request count is 16 and total response count is 137.

Each request records its group ID, source partition, representative IDs and artifact hash, source length summary, requested count, prompt and prompt SHA256, backend identity, and `uses_target_behavior=false`. No request contains representatives from another group.

Per-group allowed lengths are derived only from that group's source representatives:

`allowed_min = max(2, floor(source q10))`

`allowed_max = min(10, max(allowed_min + 2, ceil(source q90)))`

The prompt requires length variation inside those bounds, varied openings/endings, no fixed templates, and only legal device/action pairs. Historical GPT-4o lengths are not an input to this calculation.

## Response production and validation

`codex_authored_plan.json` contains 137 newly authored behavior plans. `materialize-authored` is intentionally a transparent serializer: it expands the explicit day/start-bin/action plans into schema-valid JSONL and performs no sampling or hidden generation. Every request is materialized independently.

Validation rejects missing or extra request IDs, group mismatches, wrong group counts, illegal device/action pairs, out-of-bound lengths, duplicate IDs, and within/cross-group duplicate sequences. The accepted set has 137 sequences, zero illegal actions, zero exact duplicates, and zero replacements.

## Pre-target gates

The first gate uses source representative and generated-internal statistics only: legality, exact and cross-group duplication, length variation, unique ratio, source-relative bigram/trigram entropy, template concentration, source exact copies, and maximum action share. The second gate is applied after deterministic whole-sequence train/validation splitting and detector training. It requires no exact split overlap, non-numerical-zero validation loss, validation dispersion, fewer than half near-zero losses, and full per-epoch sample coverage.

Both gates passed before target evaluation. The freeze manifest is `outputs/codex_generation_v2/fr/spring/baseline/replicate_1/pre_target_freeze_manifest.json`.

## A1 versus A2

A1 concatenated all 305 representatives into one prompt and then split only the requested output count into seven batches. A2 restores the 15 group contexts, uses group-local length limits, validates group identity, and freezes the detector before final evaluation. A1 remains an engineering smoke test. A2 improves protocol fidelity and removes action-template collapse, but the different backend means it still cannot be called an official generation reproduction.

