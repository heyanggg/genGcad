# Generation protocol v4 preregistration

A4 is the **GPT-5.6 Codex-Agent-Authored Grouped SmartGen-Compatible Baseline**. The active Codex agent directly
authors every candidate after reading each frozen prompt. Python may validate and select, but may not create or
modify event content. Candidate oversampling is prospectively fixed at 1.0, so all 137 hard-valid candidates must
be selected; there is no favorable subset search.

The protocol preserves 15 official SPPC groups, 16 requests, the historical 137 allocation, zero target behavior,
GCAD disabled, and ranking disabled. Provenance, legality/copy, distribution, semantic-v2, and all three split
feasibility checks must pass before TOF. TOF is followed by the same semantic and split checks before the frozen
reconstruction-health gate. Any failure stops before target evaluation.
