# GCAD v2 source prediction gate

The formal task is legal device-action history to the next legal device-action position. The primary metric, frozen before training, is source-validation macro F1. A selected Mixer must beat the best simple baseline on at least two of three fixed seeds and in the three-seed mean; it must also avoid all-zero collapse, obtain nonzero rare-channel recall, and avoid a severe train-validation macro-F1 gap.

All-zero, frequency, persistence, first-order Markov, n-gram-2, n-gram-3, tiny Mixer, and small Mixer use identical source splits, channels, targets, thresholding, and sparse multilabel metrics. With formal history 2, the requested n-gram-3 has only two available positions and therefore degenerates to n-gram-2; this limitation is reported rather than hidden.

Source-validation selection chose small Mixer with positive-weighted BCE. Its macro F1 was 0.388511/0.389241/0.462772, versus 0.593182/0.433730/0.546866 for the best simple n-gram baseline. It won 0/3 seeds; its mean was 0.413508 versus 0.524593. It did not collapse to all-zero and rare recall was positive, but predictive superiority failed. Formal relation extraction, fusion, and B5 are blocked.
