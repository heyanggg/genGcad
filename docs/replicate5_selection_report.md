# Replicate-5 deterministic selection report

The frozen support-preserving selector deleted 23 complete candidates without modifying event content and produced the original 137 group quotas. The selected response SHA256 is `df5a8d8a440da12c3af9207df5eba5021b850e4a7fa5694bd70276b2e8dc1dcc`; the selection trace SHA256 is `ce90af48f726e75a5d5ee48a1342b9b2fe939265b5de5943a19e7680cea50af5`.

All final constraints passed: exact group quotas, source vocabulary recall 1.0, all 39 actions supported by at least five independent sequences, zero metadata-only tokens, required length/device coverage, transition coverage `0.5439330544`, and the frozen distribution constraints. No target result, anomaly score, or reconstruction loss entered selection.

The deleted sequence IDs, in deterministic deletion order, were:

`r5_g11_005`, `r5_g21_001`, `r5_g31_004`, `r5_g41_023`, `r5_g42_006`, `r5_g01_006`, `r5_g20_004`, `r5_g40_003`, `r5_g42_003`, `r5_g61_006`, `r5_g00_009`, `r5_g41_022`, `r5_g41_006`, `r5_g41_003`, `r5_g51_003`, `r5_g60_006`, `r5_g50_001`, `r5_g11_011`, `r5_g41_012`, `r5_g30_007`, `r5_g50_008`, `r5_g10_001`, and `r5_g10_006`.

Each deletion reason is recorded as the highest frozen deterministic removal priority among candidates whose removal preserved every constraint; the complete priority values and post-deletion checks are in `selection_trace.jsonl`.
