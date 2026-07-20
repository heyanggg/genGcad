# Source-only GCAD design

The formal input is the complete TSS `split_trn.pkl`, never the frequency-distorting SSC subset. `SourceEventTensorizer` converts each flat event sequence independently to a deterministic device or device-action tensor, bins simultaneous events using binary/count semantics, serializes vocabulary/masks/hashes, and creates no cross-sequence windows. `SequenceWindowDataset` maps a history of length `H` to the next multi-hot channel vector.

`SourceGCADMixer` adapts GCAD/TSMixer's temporal mixing, feature MLP, and two residual additions. It emits one logit per channel and retains `L[t,j]=BCEWithLogits(y_hat[t,j],y[t,j])`. Early stopping and deterministic train/source-validation splits are implemented by `trainer.py`; persistence, first-order Markov, and configurable n-gram comparisons are in `prediction_baselines.py`.

For every output `j`, `gradient_relation.py` computes `|∂L_j/∂X_lag,i|`, accumulates safely across batches, and preserves `[lag,input,output]`. Mean/median sample aggregation and max/sum lag aggregation are configurable. `asymmetric_filter.py` forms `max(0,A[i,j]-A[j,i])`, clears self-loops, thresholds, normalizes, and applies per-source top-k.

Across seeds/bootstrap/partitions, `stability_filter.py` records occurrence, strength mean/std, direction consistency, rank stability, primary lag, seed count, and split count. The implemented score is:

`stable_score = occurrence_rate × mean_strength × rank_stability`.

Only edges passing all source-only thresholds reach `gss_fusion.py`. The main mode `rerank_existing` computes `(1-alpha)×normalized_GSS + alpha×normalized_GCAD` only for existing legal GSS edges; `alpha=0.2`, no new edge. `prompt_adapter.py` labels them predictive source patterns—not observed target facts or physical causality. After original TOF, `sequence_ranking.py` emits scores/ranks/weights without deleting any sequence; the downstream detector uses these weights through deterministic weighted sampling.

`mechanism_controls.py` supplies deferred E/F interfaces: deterministic edge-count/source-out-degree-matched random direction and reverse-completed symmetric relations. These controls were implemented and tested but not generated/evaluated in the first cell.
