# O. Official synthetic downstream sanity check

This group validates the downstream environment only. It is neither a new generation method nor a participant in GCAD comparisons.

Input: `anomaly_detection_pipeline/synthetic_data/fr_spring_generation_SPPC_th=0.918_gpt-4o_seq_filter_true.pkl`, SHA256 `3c2f8bf3d3233afe5bc54be37b4adf20a8565fc42419658bd8b0db7cc9a05666`, 125 sequences. The detector was trained from scratch for 15 epochs with seed 2024 on CPU/PyTorch 2.13.0+cu130. The generated-validation 95.5th percentile threshold was 1.4873873705.

| Result | Precision | Recall | F1 |
|---|---:|---:|---:|
| Official recorded | 0.763158 | 0.988636 | 0.861386 |
| Current environment | 0.737288 | 0.988636 | 0.844660 |
| Absolute difference | 0.025870 | 0 | 0.016726 |

Current accuracy is 0.818182, FPR 0.352273, and confusion matrix TN=57, FP=31, FN=1, TP=87. Normal score mean/median are 1.436672/0.00003159; attack score mean/median are 4.700089/4.471928. Full per-sequence scores remain in `outputs/codex_generation_v2/fr/spring/official_synthetic_sanity/downstream_evaluation.json`; compact evidence is under `experiment_artifacts/fr_spring_baseline_v2/official_synthetic_sanity/`.

The result supports “basic downstream reproducibility” but not bitwise equivalence. CPU execution, PyTorch build, and the unavailable original training environment plausibly explain the residual difference.
