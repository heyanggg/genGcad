# Exploratory GCAD prompt ablation

This is a single-run exploratory comparison, not a causal or multi-replicate
conclusion. Codex CLI did not expose a controllable generation sampling seed, so model
sampling variance remains a confounder.

Both runs used the same code commit, FR winter-to-spring source data, TSS, SSC/SPPC,
GSS, threshold `0.918`, experiment seed `2024`, `gpt-5.6-sol`, `reasoning=none`, and
the `environment-aware` prompt profile. The same GCAD artifact fingerprint
`842d1460e682229299c8c51802dfb98957e0051d4ac844e5b141d4b38e3ad220` contained five
stable relationships.

The 16 prompt pairs were checked directly. Each pair differed only by the inserted
GCAD directional-guidance block.

| Metric | GCAD require | GCAD off | Difference (on - off) |
| --- | ---: | ---: | ---: |
| Raw generated sequences | 196 | 219 | -23 |
| TOF final sequences | 187 | 198 | -11 |
| Detection threshold | 0.574918 | 1.002040 | -0.427122 |
| TP | 88 | 72 | +16 |
| TN | 81 | 88 | -7 |
| FP | 7 | 0 | +7 |
| FN | 0 | 16 | -16 |
| Recall | 1.000000 | 0.818182 | +0.181818 |
| Precision | 0.926316 | 1.000000 | -0.073684 |
| Accuracy | 0.960227 | 0.909091 | +0.051136 |
| F1 | 0.961749 | 0.900000 | +0.061749 |

The observed direction supports the hypothesis that GCAD guidance can improve attack
coverage in this environment: the require run removed all 16 false negatives seen in
the off run, at the cost of seven false positives. This result must remain labeled
exploratory until generation replicates are performed.

TOF candidate scratch files were deleted by the pre-archive default in these two runs;
their archive manifests explicitly record that limitation. The merged input, first
filter, final TOF data, TOF model, detector split, detector model, metrics, exact
prompts, responses, source snapshot, SSC/GSS, and GCAD artifact are preserved.
