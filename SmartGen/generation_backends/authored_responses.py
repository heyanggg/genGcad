from __future__ import annotations

import json
from pathlib import Path

from .frozen_protocol import verify_frozen_requests
from .response_loader import load_jsonl


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def materialize_authored_responses(requests_path: str | Path, authored_plan_path: str | Path, output_path: str | Path):
    """Serialize explicitly agent-authored plans; this performs no generation or random sampling."""
    requests_path = Path(requests_path)
    if "baseline_source_semantic" in requests_path.parts and requests_path.parent.name == "replicate_2":
        verify_frozen_requests(requests_path)
    requests = load_jsonl(requests_path)
    request_map = {item["request_id"]: item for item in requests}
    plan = json.loads(Path(authored_plan_path).read_text(encoding="utf-8"))
    if {item["request_id"] for item in plan} != set(request_map):
        raise ValueError("authored plan request IDs must exactly match exported requests")
    responses = []
    for record in plan:
        request = request_map[record["request_id"]]
        if record["group_id"] != request["group_id"]:
            raise ValueError("authored group does not match request")
        if len(record["sequences"]) != request["requested_sequence_count"]:
            raise ValueError("authored sequence count does not match request")
        sequences = []
        for sequence in record["sequences"]:
            day_index = int(sequence["day"])
            start_bin = int(sequence["start_bin"])
            events = []
            for offset, action_spec in enumerate(sequence["actions"]):
                device, action = action_spec.split("|", 1)
                absolute_bin = start_bin + offset
                event_day = DAYS[(day_index + absolute_bin // 8) % 7]
                hour_bin = absolute_bin % 8
                events.append({
                    "day": event_day,
                    "hour": f"({hour_bin * 3}~{hour_bin * 3 + 3})",
                    "device": device,
                    "action": action,
                })
            sequences.append({"sequence_id": sequence["sequence_id"], "events": events})
        responses.append({
            "request_id": request["request_id"],
            "experiment_id": request["experiment_id"],
            "method": "baseline_v2",
            "group_id": request["group_id"],
            "generation_backend": "codex_agent_file",
            "generation_batch": request["generation_batch"],
            "sequences": sequences,
            "generation_notes": {
                "used_target_behavior": False,
                "used_target_labels": False,
                "copied_from_existing_synthetic_data": False,
                "authored_by_current_codex_agent": True,
                "serializer_is_non_generative": True,
            },
            "schema_version": "1.0",
        })
    Path(output_path).write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in responses), encoding="utf-8"
    )
    return responses
