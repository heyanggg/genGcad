# Replicate-5 gate results

The formal gate sequence reached reconstruction health and stopped there:

1. provenance: passed;
2. candidate JSON/schema/static legality: passed after six permitted hard-invalid replacements;
3. candidate duplicate/source-copy: passed after those replacements;
4. candidate support-plan audit: passed;
5. deterministic selection: passed, 160 to 137;
6. selected generation distribution: passed;
7. selected pre-TOF `source_semantic_v2`: passed;
8. pre-TOF `split_feasibility_v1`: all seeds 2024/2025/2026 passed;
9. formal SmartGen TOF: ran on CPU, 137 input and 137 final;
10. post-TOF `source_semantic_v2`: passed;
11. post-TOF `split_feasibility_v1`: all three seeds passed;
12. `reconstruction_health_v1`: failed;
13. target final evaluation: not run.

Pre- and post-TOF semantic metrics were source action precision `1.0`, global vocabulary recall `1.0`, group-aware recall `0.9863492063`, weighted recall `1.0`, rare-action recall `1.0`, transition coverage `0.5439330544`, metadata-only token share `0`, and minimum independent sequence support `6`. The formal minimum remained four.

The first TOF invocation unexpectedly used a newly visible CUDA device and is preserved only under `tof_invalid_cuda_attempt/`; it is not a formal result. The task-frozen formal TOF was rerun with CUDA hidden and reports `device=cpu`. It retained 131 at stage 1, recovered all six outliers at stage 2, and finalized 137.

Target normal, target attack, and target labels were never read. GCAD and Ranking remained disabled.
