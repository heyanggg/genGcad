# Source semantic v2

Protocol v5 does not alter this gate. Minimum independent sequence support remains four and the maximum low-support action ratio remains zero. Pool support six and selected support five are prospective generation safety margins, not formal threshold changes.

`source_semantic_v2` retains v1 and adds source-vocabulary recall, frequency-weighted recall, rare-action recall,
static-metadata-only usage, per-action sequence support, and transition support. Its source denominator is only
`source_observed_target_legal_actions`: actions observed in source-normal behavior and legal in the target static
device/action mapping. Static metadata coverage is diagnostic and is never substituted for source recall.

The v4 thresholds are source-only and frozen before A4 response authorship. The design was motivated by the
replicate-3 post-hoc diagnosis, so A4 is a prospective test rather than a repair of replicate-3.

Replicate-5 passed this unchanged gate both before and after formal CPU TOF. Its minimum per-action sequence support was six, global/weighted/rare recalls were 1.0, group-aware recall was approximately 0.98635, and transition coverage was approximately 0.54393.
