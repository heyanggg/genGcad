# Action support and split feasibility

`split_feasibility_v1` uses deterministic coverage-constrained multi-label stratification over SPPC group,
length bin, action set, rare actions, and device set. Seeds 2024, 2025, and 2026 are all mandatory. Validation
actions must have at least three distinct training-sequence supports; unseen and low-support validation actions
must both be zero. No favorable seed may be selected.

The split changes only membership. The official action-only detector, Transformer architecture, loss, data-loader
shuffle setting, 15 epochs, and 95.5% threshold percentile remain unchanged.
