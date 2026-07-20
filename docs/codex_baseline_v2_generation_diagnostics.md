# A2 generation and reconstruction diagnostics

All generation gates below were computed before target behavior was opened. A1 statistics are post-hoc diagnostics of its saved action templates; they were not used to tune A2 against target metrics.

| Metric | A1 smoke | A2 grouped |
|---|---:|---:|
| Sequences | 137 | 137 |
| Length min/median/max | 4/4/5 | 2/4/8 |
| Length distribution | 4:116, 5:21 | 2:33, 3:33, 4:30, 5:15, 6:12, 7:8, 8:6 |
| Unique action-template ratio | 0.1460 | 1.0000 |
| Repeated action templates | 117 | 0 |
| Near-duplicate pairs (similarity ≥0.8) | not recorded in v1 | 0 |
| Unique unigram/bigram/trigram | 50/58/43 | 145/290/238 |
| Normalized bigram/trigram entropy | 0.9914/0.9996 | 0.9768/0.9927 |
| Top-1/top-5 template share | 0.0511/0.2555 | 0.00730/0.03650 |
| Effective sequence count | 19.97 | 137.00 |
| Opening/ending templates | 20/18 | 113/119 |
| Maximum action share | 0.05975 | 0.04851 |

Entropy is normalized over observed n-grams, so A1 can have high normalized entropy despite having only 58/43 unique bigrams/trigrams and 20 unique action templates. The unique counts, effective count, and template shares expose the actual A1 collapse. A2 covers 29 devices and 145 legal device/action values (66.21% of static legal vocabulary), has no cross-group duplicates, and exactly copies no source representative. The source-only generation gate passed.

Original two-stage TOF retained 121/137 at stage one, recovered 15/16 outliers at stage two, and produced 136 sequences. Whole-sequence deterministic splitting produced 108 train and 28 validation sequences with zero exact overlap and zero duplicate action templates.

## Reconstruction behavior

Training loss decreased from 5.3404 to 0.000327 across 15 epochs. Validation losses are strongly bimodal: 11/28 are at or below `1e-4`, 19/28 at or below `0.1`, while 9/28 exceed `1.0`. Summary: min `6.62e-6`, median `2.255e-4`, mean `0.9269`, P90 `3.0642`, P95.5/threshold `4.2920`, max `5.4180`, standard deviation `1.5424`. This is not A1's across-the-board numerical-zero collapse, but it shows that the broad generated combinations split into very easy and poorly reconstructed modes.

Median validation loss by length was: length 2 `7.12e-5` (n=7), length 3 `2.16e-4` (n=7), length 4 `3.22e-4` (n=7), length 5 `1.756` (n=3), length 6 `6.62e-6` (n=1), length 7 `1.090` (n=1), and length 8 `7.53e-5` (n=2). Small counts prevent a monotonic length conclusion. Mean global action frequency versus validation loss had Pearson correlation `-0.291`; common actions alone therefore do not explain the high-loss mode. Exact train/validation leakage and template duplication are both zero.

The reconstruction gate passed because its fixed, predeclared numerical-collapse checks all passed. Passing that gate does not assert that the generated distribution matches the official synthetic distribution or guarantees target performance.

## Frozen final evaluation and failure analysis

After freezing, the one permitted target evaluation produced threshold 4.2920, precision 0.7647, recall 0.4432, accuracy 0.6534, F1 0.5612, and FPR 0.1364 (TN=76, FP=12, FN=49, TP=39). Target normal median/P90 scores were 0.000280/4.2934; attack median/P90 scores were 3.2847/6.0508. The threshold is above the attack median, explaining the 49 false negatives and low recall.

Thus A2 fixed A1's protocol and template-collapse defects but overcorrected toward heterogeneous combinations that the small detector split reconstructs bimodally. This is why the credible engineering checks can pass while downstream F1 gets worse. The result was not used to regenerate A2, choose another batch, alter the threshold, or tune a gate.

