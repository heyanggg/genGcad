# Codex GPT-5.6 agent generation definition

In protocol v5, the same authorship boundary applies to all 160 candidates. Support targets guide a complete request batch but never specify a particular sequence's actions. Python may count, validate, and delete whole candidates; it cannot create, replace, or modify events.

The formal v4 backend is `codex_gpt56_agent_file`. Python extracts source data, builds and freezes complete
SmartGen-compatible prompts, validates response artifacts, selects only from a prospectively frozen candidate
pool, converts validated responses, and runs gates. Python does not create, complete, reorder, or rewrite event
content.

The active Codex agent, using the user-specified GPT-5.6 model family, reads each frozen prompt and authors each
candidate behavior sequence directly into the raw response artifact. No external OpenAI API, API key, HTTP
request, fabricated request ID, token count, temperature, sampling seed, or SDK metadata is used.

Formal provenance metadata is:

```json
{
  "generation_backend": "codex_gpt56_agent_file",
  "content_author": "codex_gpt56_agent",
  "external_api_used": false,
  "api_key_used": false,
  "programmatic_event_construction": false,
  "target_behavior_used": false
}
```

Any response produced by an authored-plan builder, template loop, random event generator, GSS constructor, or
source-vocabulary combinator is ineligible for this backend and must fail provenance before later gates run.

Replicate-5 followed this boundary for all 160 candidates. Six hard-invalid candidates were directly re-authored by Codex within the frozen replacement allowance; Python only serialized and verified the exact six-ID replacement mapping.
