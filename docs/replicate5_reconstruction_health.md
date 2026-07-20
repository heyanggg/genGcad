# Replicate-5 reconstruction health

`reconstruction_health_v1` ran only after every generation, semantic, split, and formal CPU TOF gate passed. It used the unchanged policy SHA256 `600c1ca86cc7e0b412521b1cc525ddd66fc3e21ff999f938a79ddc029a4a822c`, fixed split seeds 2024/2025/2026, 15 training epochs, model seed 2024, and the 95.5th generated-validation percentile. It did not use the official synthetic threshold or target behavior.

The thresholds were:

- seed 2024: `0.00030860098806442695`;
- seed 2025: `0.00025418866054678803`;
- seed 2026: `0.0001469389228441287`.

All splits had 109 train and 28 validation sequences, zero exact overlap, zero near-zero-loss ratio, sufficient p90-p10 dispersion, and no frozen bimodality failure. However, the top 10% validation-loss contributions were `0.7033255`, `0.7564080`, and `0.7887412`, all above the frozen maximum `0.5`. Threshold coefficient of variation was `0.2838959` above `0.2`, and relative range was `0.6359924` above `0.5`.

The formal reconstruction-health gate therefore failed on both per-split high-loss dominance and cross-split threshold instability. No threshold, split, percentile, or policy was changed after observing these losses.
