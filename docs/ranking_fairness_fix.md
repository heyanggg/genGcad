# Ranking fairness fix

## Defect

Ranking v1 used `WeightedRandomSampler(replacement=True)`. Even when every relationship weight was equal, an epoch could repeat some sequences and omit others. Ranking-only therefore did not share the baseline's data coverage or batch order, so its F1 decline was primarily confounded by the replacement sampler rather than isolating a relationship-weight effect.

## Formal path

The detector now always uses the same ordinary, non-shuffled DataLoader and deterministic train/validation split. Every training sequence appears exactly once per epoch. Relationship scores are converted to mean-one weights clipped to `[0.9, 1.1]`, and weighting is applied to whole-sequence masked reconstruction losses:

`sum_i(w_i * token_loss_sum_i) / sum_i(w_i * valid_token_count_i)`

This preserves the original global masked-loss definition when all weights equal one. It also keeps the full training set when weights are non-uniform.

If all relation scores are equal or their coefficient of variation is below the fixed numerical tolerance, ranking bypasses the weighted path, records `ranking_signal_absent=true`, and runs the baseline training path. Reports include min/max/mean/CV, entropy, effective sample size, policy name, and full-set coverage.

`WeightedRandomSampler` is absent from the formal downstream path. No C/D experiment was rerun in this baseline-repair phase, so the old C/D metrics remain historical smoke results and are not reinterpreted as evidence for GCAD.

## Equivalence evidence

Automated tests check identical DataLoader/batch order, all-sample coverage, exact unit-weight loss equality, identical first update under fixed seed, uniform-score bypass, no replacement sampler, bounded mean-one weights, unchanged split membership, non-uniform full coverage, gradients for every sample, and ESS/entropy/CV reporting. The full suite passes on CPU; the only skip is the honest CUDA-only test.

