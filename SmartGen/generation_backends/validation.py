from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .response_loader import load_jsonl
from .schemas import SCHEMA_VERSION, ValidationFailure, sequence_fingerprint


def _event_failure(event, metadata):
    required = {"day", "hour", "device", "action"}
    if not isinstance(event, dict) or not required.issubset(event):
        return "schema", "event fields day/hour/device/action are required"
    if not isinstance(event["day"], str) or not isinstance(event["hour"], str):
        return "schema", "day and hour must be strings"
    if not isinstance(event["device"], str) or not isinstance(event["action"], str):
        return "schema", "device and action must be strings"
    device = event["device"]
    if device not in metadata:
        return "illegal_device", device
    action = event["action"].split(":", 1)[-1]
    allowed = [item.split(":", 1)[-1] for item in metadata[device]]
    if action not in allowed:
        return "illegal_action", f"{device}:{action}"
    return None


def validate_responses(
    requests_path: str | Path,
    responses_path: str | Path,
    output_dir: str | Path,
    source_fingerprints: set[str] | None = None,
    raise_on_failure: bool = True,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures: list[ValidationFailure] = []
    try:
        requests = load_jsonl(requests_path)
    except ValueError as exc:
        raise ValueError(f"request file invalid: {exc}") from exc
    try:
        responses = load_jsonl(responses_path)
    except ValueError as exc:
        failures.append(ValidationFailure(None, None, "json_or_markdown", str(exc)))
        responses = []
    request_map = {item["request_id"]: item for item in requests}
    response_counts = Counter(item.get("request_id") for item in responses)
    for request_id in request_map:
        if response_counts[request_id] == 0:
            failures.append(ValidationFailure(request_id, None, "missing_response", "no response"))
        elif response_counts[request_id] > 1:
            failures.append(ValidationFailure(request_id, None, "extra_response", "more than one response"))
    for request_id in response_counts:
        if request_id not in request_map:
            failures.append(ValidationFailure(request_id, None, "extra_response", "unknown request_id"))

    fingerprints: dict[str, str] = {}
    valid_responses = []
    total_initial = 0
    for response in responses:
        request_id = response.get("request_id")
        request = request_map.get(request_id)
        if request is None or response_counts[request_id] != 1:
            continue
        if response.get("schema_version") != SCHEMA_VERSION or response.get("generation_backend") != "codex_agent_file":
            failures.append(ValidationFailure(request_id, None, "schema", "backend or schema_version mismatch"))
            continue
        notes = response.get("generation_notes", {})
        if any(notes.get(key) is not False for key in ("used_target_behavior", "used_target_labels", "copied_from_existing_synthetic_data")):
            failures.append(ValidationFailure(request_id, None, "boundary", "generation notes do not assert source-only generation"))
            continue
        sequences = response.get("sequences")
        if not isinstance(sequences, list):
            failures.append(ValidationFailure(request_id, None, "schema", "sequences must be a list"))
            continue
        total_initial += len(sequences)
        if len(sequences) != request["requested_sequence_count"]:
            failures.append(ValidationFailure(request_id, None, "sequence_count", f"expected {request['requested_sequence_count']}, got {len(sequences)}"))
        response_valid = True
        local_ids = set()
        constraints = request["sequence_constraints"]
        for sequence in sequences:
            sequence_id = sequence.get("sequence_id") if isinstance(sequence, dict) else None
            if not sequence_id or sequence_id in local_ids:
                failures.append(ValidationFailure(request_id, sequence_id, "schema", "missing or duplicate sequence_id"))
                response_valid = False
                continue
            local_ids.add(sequence_id)
            events = sequence.get("events")
            if not isinstance(events, list) or not events:
                failures.append(ValidationFailure(request_id, sequence_id, "schema", "events must be a non-empty list"))
                response_valid = False
                continue
            if not constraints["min_events"] <= len(events) <= constraints["max_events"]:
                failures.append(ValidationFailure(request_id, sequence_id, "length", str(len(events))))
                response_valid = False
            for event in events:
                issue = _event_failure(event, request["target_static_device_metadata"])
                if issue:
                    failures.append(ValidationFailure(request_id, sequence_id, issue[0], issue[1]))
                    response_valid = False
            fingerprint = sequence_fingerprint(events)
            if fingerprint in fingerprints:
                failures.append(ValidationFailure(request_id, sequence_id, "duplicate", f"duplicates {fingerprints[fingerprint]}"))
                response_valid = False
            elif source_fingerprints and fingerprint in source_fingerprints:
                failures.append(ValidationFailure(request_id, sequence_id, "source_copy", "exact source sequence copy"))
                response_valid = False
            fingerprints[fingerprint] = sequence_id
        if response_valid and len(sequences) == request["requested_sequence_count"]:
            valid_responses.append(response)

    failure_records = [failure.to_dict() for failure in failures]
    (output / "generation_failures.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failure_records), encoding="utf-8"
    )
    (output / "generation_responses_validated.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in valid_responses), encoding="utf-8"
    )
    replacement_path = output / "replacement_mapping.json"
    if not replacement_path.exists():
        replacement_path.write_text("[]\n", encoding="utf-8")
    counts = Counter(item["category"] for item in failure_records)
    final_valid = sum(len(item["sequences"]) for item in valid_responses)
    report = {
        "request_count": len(requests),
        "initial_sequence_count": total_initial,
        "json_failures": counts["json_or_markdown"],
        "schema_failures": counts["schema"],
        "illegal_device_count": counts["illegal_device"],
        "illegal_action_count": counts["illegal_action"],
        "length_failures": counts["length"],
        "duplicate_count": counts["duplicate"],
        "missing_response_count": counts["missing_response"],
        "extra_response_count": counts["extra_response"],
        "repair_count": 0,
        "final_valid_sequence_count": final_valid,
        "failure_count": len(failures),
        "external_api_called": False,
        "api_key_used": False,
        "target_behavior_read": False,
    }
    (output / "generation_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures and raise_on_failure:
        raise ValueError(f"generation validation failed with {len(failures)} issue(s)")
    return report


def save_replacement_mapping(output_dir: str | Path, mapping: list[dict]) -> None:
    Path(output_dir, "replacement_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
