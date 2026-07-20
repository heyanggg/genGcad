# Replicate-4 results

Formal result: **prospective replicate-4 source_semantic_v2 failure**.

The 137-sequence set covered all 39 source-observed target-legal actions and used no metadata-only actions. Source action precision, global vocabulary recall, weighted recall, and rare-action recall were all `1.0`; group-aware recall was `0.9897435897`; source transition coverage was `0.8364779874`; unseen transition ratio was `0.1635220126`.

The frozen minimum sequence support was four. `AirPurifier:setAirPurifierMode` and `Television:setPictureMode` each appeared six times but in only three distinct sequences. Consequently, minimum sequence support was three and the low-support action ratio was `2/39 = 0.0512820513`, versus the frozen maximum `0.0`.

The failure conclusion is immutable for replicate-4. Split feasibility, TOF, post-TOF checks, reconstruction health, target evaluation, GCAD, and Ranking were not run. Target normal, attack, and label data were not read.
