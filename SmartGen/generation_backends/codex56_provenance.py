from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .response_loader import load_jsonl


BACKEND = "codex_gpt56_agent_file"
AUTHOR = "codex_gpt56_agent"
FORBIDDEN_REQUEST_FIELDS = {
    "authored_plan", "authored_plan_path", "event_template", "per_sequence_actions",
    "programmatic_events", "deterministic_event_plan",
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_codex56_requests(requests: list[dict]) -> None:
    if not requests:
        raise ValueError("formal Codex request set is empty")
    for request in requests:
        if request.get("generation_backend") != BACKEND:
            raise ValueError("formal request backend is not codex_gpt56_agent_file")
        if request.get("model_family") != "GPT-5.6 via Codex agent":
            raise ValueError("formal request model family is missing")
        if FORBIDDEN_REQUEST_FIELDS & set(request):
            raise ValueError("formal Codex request contains a programmatic event plan")
        boundary = request.get("data_boundary", {})
        if any(boundary.get(key) is not False for key in (
            "uses_target_behavior", "uses_target_normal", "uses_target_attack", "uses_target_labels"
        )):
            raise ValueError("formal request violates zero-target boundary")


def validate_codex56_response_metadata(response: dict) -> None:
    expected = {
        "generation_backend": BACKEND,
        "content_author": AUTHOR,
        "external_api_used": False,
        "api_key_used": False,
        "programmatic_event_construction": False,
        "target_behavior_used": False,
    }
    for key, value in expected.items():
        if response.get(key) != value:
            raise ValueError(f"formal Codex response provenance mismatch: {key}")
    notes = response.get("generation_notes", {})
    if notes.get("authored_plan_used") is not False or notes.get("python_events_constructed") is not False:
        raise ValueError("programmatic event construction cannot be marked as Codex generation")


def apply_generation_provenance_gate(directory: str | Path) -> dict:
    directory = Path(directory)
    requests = load_jsonl(directory / "generation_requests.jsonl")
    validate_codex56_requests(requests)
    if (directory / "codex_authored_plan.json").exists():
        raise ValueError("formal Codex directory contains an authored-plan artifact")
    attestation_path = directory / "agent_generation_attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    raw_path = directory / "generation_responses_raw.jsonl"
    checks = {
        "backend": attestation.get("generation_backend") == BACKEND,
        "author": attestation.get("content_author") == AUTHOR,
        "direct_agent_authorship": attestation.get("direct_agent_authorship") is True,
        "external_api_unused": attestation.get("external_api_used") is False,
        "api_key_unused": attestation.get("api_key_used") is False,
        "programmatic_event_construction_absent": attestation.get("programmatic_event_construction") is False,
        "target_behavior_unused": attestation.get("target_behavior_used") is False,
        "raw_response_sha256": attestation.get("generation_responses_raw_sha256") == sha256_file(raw_path),
        "request_sha256": attestation.get("generation_requests_sha256")
        == sha256_file(directory / "generation_requests.jsonl"),
    }
    gate = {
        "stage": "codex_gpt56_generation_provenance",
        "passed": all(checks.values()),
        "checks": checks,
        "generation_backend": BACKEND,
        "content_author": AUTHOR,
        "external_api_used": False,
        "api_key_used": False,
        "programmatic_event_construction": False,
        "uses_target_behavior": False,
    }
    (directory / "generation_provenance_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    if not gate["passed"]:
        raise ValueError("GPT-5.6 Codex generation provenance gate failed")
    return gate
