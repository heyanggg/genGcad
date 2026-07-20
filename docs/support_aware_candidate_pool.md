# Support-aware candidate pool

The support plan contains only group-level targets. It does not contain events, sequence-specific action assignments, templates, rotations, formulas for event composition, or an authored plan. The active Codex GPT-5.6 agent remains responsible for every candidate event.

Candidate support counts are based on distinct sequence IDs. Repeating an action multiple times inside one candidate contributes one unit. Candidate support-plan failure or deterministic-selection infeasibility freezes replicate-5 as failed; Python may not repair candidates or request more generations.
