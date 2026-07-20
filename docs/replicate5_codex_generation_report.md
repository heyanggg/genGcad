# Replicate-5 Codex generation report

Replicate-5 used the frozen A5 backend `codex_gpt56_agent_file`. The active Codex GPT-5.6 agent directly authored all 160 candidates across 16 requests; Python did not construct or modify events, no external API or API key was used, no replicate-4 candidate was reused, and no target behavior or labels were read.

Candidate counts by source SPPC group were `0_0=9`, `0_1=7`, `1_0=14`, `1_1=12`, `2_0=8`, `2_1=7`, `3_0=7`, `3_1=9`, `4_0=9`, `4_1=34`, `4_2=11`, `5_0=11`, `5_1=7`, `6_0=8`, and `6_1=7`. These are the frozen largest-remainder quotas and total 160.

The first hard-validation pass found five exact candidate duplicates and one exact source-representative copy. All six were permitted hard-invalid categories and were directly re-authored by Codex, below the frozen maximum of ten. The final pool had zero JSON, schema, provenance, device, action, length, duplicate, source-copy, missing-response, or count errors. Initial raw failures and the six-item replacement mapping are preserved.

The final raw response SHA256 is `520452a96773309f4430c7fd8b93892fd06c4bfdd3ab91d69e12161c18d1a674`. The frozen request SHA256 is `e74c63ec931b8bd75c02b1a95e40dca6c2094122e549428784d4bde2dd85bc13`.

The candidate support audit passed. Every one of the 39 source-observed target-legal actions had independent sequence support of at least six, every group allocation was met, and no source action appeared in an unsupported source group.
