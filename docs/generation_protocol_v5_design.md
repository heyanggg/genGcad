# Generation protocol v5 design

Protocol v5 defines A5, the Support-Aware Codex GPT-5.6 Candidate-Pool Baseline. It keeps every formal v4 threshold unchanged. The change is entirely prospective: the active Codex agent authors a fixed pool of 160 candidates, and Python deterministically deletes 23 candidates without changing event content.

Candidate group quotas use largest remainder from the original 137 group quotas. All 39 source actions receive the same pool support target of six and selected-set target of five. These are safety margins; the formal semantic minimum remains four. Group placement is restricted to direct SPPC evidence or source-normal day evidence.

The selector freezes this priority: greatest action-support redundancy, greatest group surplus, greatest candidate similarity, least observed-transition contribution, then ascending sequence ID. It cannot use target behavior, anomaly scores, reconstruction losses, or supplemental generation.
