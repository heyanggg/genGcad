# Frozen reconstruction_health_v1 policy

`reconstruction_health_v1` was specified and hashed without observing any replicate-2 reconstruction loss. Replicate-2 failed before TOF, so this policy is not applied to that experiment. It exists to make future safety and equivalence behavior testable without opening target data.

The frozen configuration is `configs/generation_protocol/reconstruction_health_v1.json`, SHA256 `600c1ca86cc7e0b412521b1cc525ddd66fc3e21ff999f938a79ddc029a4a822c`. It uses generated-only diagnostics for fixed split seeds 2024, 2025, and 2026, with 95.5% generated-validation thresholds. It explicitly does not use the official 1.4874 threshold or any target behavior.

For every split it requires:

- at least 20 validation sequences;
- train/validation exact overlap equal to zero;
- validation loss at or below `1e-6` for no more than 25% of sequences;
- P90−P10 loss dispersion of at least `1e-4`;
- the highest-loss 10% contributing no more than 50% of total validation loss;
- no two clusters separated by at least one log10 decade when each cluster contains at least 20% of losses.

Across the three fixed splits, threshold coefficient of variation must be at most 0.2 and threshold relative range `(max−min)/median` at most 0.5. All three generation/reconstruction gate artifacts must pass and declare `uses_target_behavior=false` before the only target-evaluation function can open target files.

Automated tests use synthetic loss arrays to prove that near-zero collapse, high-loss-tail dominance, obvious bimodality, unstable split thresholds, target roles, and incomplete gate sets are rejected. These tests do not run or tune the replicate-2 model.

