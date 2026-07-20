# Zero-target-behavior data boundary

`data_roles.py` defines `source_normal`, `source_validation`, `target_metadata`, `target_normal_eval`, and `target_attack_eval`. `data_boundary.py::require_roles` validates a role before a stage opens an artifact. Tensorization accepts only source normal; train/extract/stability accept source normal/validation; fusion/prompt/generation/TOF/ranking additionally accept static target metadata; threshold accepts source data only. Target normal and attack are accepted only by `final_evaluation`.

The boundary is stricter than zero-target-label. Before final evaluation the pipeline does not read target sequences, lengths, device/action frequencies, temporal distributions, transitions, graphs, normal fine-tuning, attack labels, or target-calibrated thresholds. Permitted target metadata is limited to scenario prose and legal device→action mappings already used by SmartGen.

The generated PKL is tagged `source_normal` for detector training. The detector threshold is percentile 95.5 of its held-out generated validation set. Only after training and threshold creation does `downstream_evaluation.py` pass `target_normal_eval` and `target_attack_eval` to `evaluate`. Reports record `target_data_first_used_at=final_evaluation` and `metrics_not_used_for_selection=true`.

Tests reject both target roles at train, generation, threshold, TOF, and ranking; accept attack only at final evaluation; and exercise the file backend without networking or API keys. This is an application guard, not an OS-level information-flow proof. Raw target files existed on disk, but no pre-evaluation program path opened them. Final A/B/C/D metrics were observed only after all parameters and generated data were frozen and were not fed back.
