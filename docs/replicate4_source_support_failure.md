# Replicate-4 source-support failure

Replicate-4 is a genuine GPT-5.6 Codex-agent-authored generation. Its provenance, legality, copy-safety, and generation-distribution gates passed. It failed only because `AirPurifier:setAirPurifierMode` and `Television:setPictureMode` each occurred in three independent sequences, below the frozen `source_semantic_v2` minimum of four. Repeated tokens inside one sequence do not count as independent support.

The result is frozen by tag `replicate4-source-support-failure`. No replicate-4 response, selection, threshold, or formal conclusion may be changed.
