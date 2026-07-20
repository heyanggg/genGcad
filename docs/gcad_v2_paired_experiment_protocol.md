# GCAD v2 paired experiment protocol

The unexecuted paired protocol is frozen in `configs/gcad_source_v2/paired_experiment.yaml`. It would compare frozen A5 with a new B5 using generation protocol v5, 160 fresh Codex candidates, deterministic selection to 137, identical official detector settings, seeds 2024/2025/2026, and the official 95.5% generated-validation threshold. The only method difference would be frozen GCAD-reranked GSS in the Prompt. Reconstruction health would be reported as a diagnostic rather than used to select an arm.

The prerequisite source gate failed, so this protocol did not authorize target access or generation.
