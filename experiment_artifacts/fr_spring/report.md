# FR winter→spring smoke report

The complete source-only and final-evaluation pipeline ran on CPU. Tensorization used 1,728 TSS sequences, 40 device-action channels and 165 valid history-4 windows. Across seeds 2024/2025/2026, Mixer validation BCE was 0.01781/0.04159/0.01542; the best Markov or n-gram value was lower in every seed. Two disjoint partition/bootstrap replicates were added for stability.

Five asymmetric graphs contained 77, 50, 76, 181 and 188 edges. Stability filtering retained 39 edges across five seeds/three split IDs. Fusion saw 82 original GSS edges, influenced only two, added none, and changed no rank. This is weak evidence of useful complementary structure.

Codex authored 137 baseline and 137 GCAD-GSS sequences in seven batches each. Both passed initial validation with zero repairs/failures. No external API, API key, old synthetic data, target behavior, or target labels were used. Original two-stage TOF retained 135 baseline and 132 GCAD-GSS sequences. C reused A's TOF PKL; D reused B's; ranking was soft weighted sampling and deleted nothing.

| Group | Precision | Recall | Accuracy | F1 |
|---|---:|---:|---:|---:|
| A baseline | 0.6350 | 0.9886 | 0.7102 | 0.7733 |
| B GCAD-GSS | 0.5000 | 1.0000 | 0.5000 | 0.6667 |
| C ranking | 0.5000 | 1.0000 | 0.5000 | 0.6667 |
| D both | 0.5000 | 1.0000 | 0.5000 | 0.6667 |

The enhancement failed to improve the smoke cell. No post-test tuning or batch selection was performed. Large auditable artifacts remain at `outputs/gcad_source/fr/spring`, `outputs/codex_generation/fr/spring`, and `outputs/downstream/fr/spring`; hashes and exact metrics are in `summary.json`.
