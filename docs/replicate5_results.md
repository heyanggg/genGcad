# Replicate-5 results

Replicate-5 is frozen as a prospective `reconstruction_health_v1` failure. A5 successfully repaired replicate-4's generation-time independent sequence-support problem: 160 Codex-authored candidates passed uniform support auditing, deterministic selection found a valid 137 subset, both generation-side gates passed, all pre/post TOF split checks passed, and the formal CPU TOF retained all 137 sequences.

The experiment nevertheless failed before target access because reconstruction loss was dominated by a few validation samples for every fixed split and the three 95.5% thresholds were unstable. The result is not a target-performance result: target normal/attack behavior and labels were never read, Precision/Recall/F1/FPR were not computed, and no final threshold was selected for deployment.

Tests: `109 passed, 3 warnings`. GCAD and Ranking remained disabled. The local output root is `outputs/codex_generation_v2/fr/spring/codex_support_aware_v5/replicate_5/`, with `checksums.sha256` as the complete artifact manifest.
