# GCAD v2 relation extraction

The v2 implementation supports per-output, per-lag positive-target gradients, reverse-edge comparison, activation/co-occurrence support filtering, asymmetric margins, and cross-run stability fields. Unit tests cover these mechanisms.

Formal extraction was not run. The source prediction gate failed first, so no gradient edge can be called a formal GCAD v2 relation and no `stable_relation_v2` artifact exists. This preserves the preregistered causal order of evidence.
