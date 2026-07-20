# Replicate-4 GPT-5.6 generation report

Replicate-4 is the A4 GPT-5.6 Codex-agent-authored grouped SmartGen-compatible baseline. The active Codex agent read the 16 frozen SmartGen requests and directly authored all 137 candidate sequences. Python extracted and archived prompts, serialized agent-authored records, validated candidates, and ran source-only gates; it did not choose or construct event content.

The frozen request SHA256 is `9bedf16eb32a3fc1831017a0182ed838b57c95199509f61e88b3f5572357a419`. The final raw response SHA256 is `a18318752cf1718c3b26813aa2575f10954ab0490c28aaf69fb5519b0c9324c1`.

The initial hard validation found eight generated exact duplicates and two exact source-representative copies. These ten candidates were preserved in the initial raw artifact and replaced by the Codex agent under the frozen hard-error-only allowance. No semantic score or target result was used. Final validation selected all 137 hard-valid candidates, with zero JSON, schema, legality, duplicate, or source-copy failures.

Provenance, legality, copy safety, and the original generation-distribution gate passed. `source_semantic_v2` then failed prospectively, so no later stage ran.
