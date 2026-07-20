import json
import pickle

import pytest

from SmartGen.generation_backends.codex_file import CodexFileBackend
from SmartGen.generation_backends.response_loader import load_jsonl
from SmartGen.generation_backends.schemas import canonical_generation_method
from SmartGen.generation_backends.validation import validate_responses


METADATA = {"Light": ["switch on", "switch off"], "Blind": ["windowShade open"]}


def export(tmp_path, count=10):
    backend = CodexFileBackend()
    requests = backend.export(
        tmp_path,
        experiment_id="exp",
        dataset="fr",
        context="spring",
        method="baseline",
        replicate=1,
        prompt="complete prompt",
        requested_sequence_count=count,
        batch_size=10,
        source_sequence_count=4,
        target_context_description={"season": "spring"},
        target_static_device_metadata=METADATA,
        original_gss_path="gss.json",
    )
    return backend, requests


def response(requests, duplicate=False):
    records = []
    serial = 0
    for request in requests:
        sequences = []
        for i in range(request["requested_sequence_count"]):
            n = 0 if duplicate else serial
            sequences.append({
                "sequence_id": f"s{serial}",
                "events": [
                    {"day": "Monday", "hour": f"({(n // 8) * 3}~{(n // 8) * 3 + 3})", "device": "Light", "action": "switch on"},
                    {"day": "Monday", "hour": f"({(n % 8) * 3}~{(n % 8) * 3 + 3})", "device": "Light", "action": "switch off"},
                ],
            })
            serial += 1
        records.append({
            "request_id": request["request_id"], "experiment_id": "exp", "method": "baseline",
            "generation_backend": "codex_agent_file", "generation_batch": request["generation_batch"],
            "sequences": sequences,
            "generation_notes": {"used_target_behavior": False, "used_target_labels": False, "copied_from_existing_synthetic_data": False},
            "schema_version": "1.0",
        })
    return records


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def test_export_is_deterministic_and_archives_full_prompt(tmp_path):
    _, first = export(tmp_path / "a")
    _, second = export(tmp_path / "b")
    assert first == second
    assert (tmp_path / "a/prompt_archive" / f"{first[0]['request_id']}.txt").read_text() == "complete prompt"
    assert first[0]["data_boundary"]["uses_target_behavior"] is False


def test_fairness_canonical_generation_groups():
    assert canonical_generation_method("baseline") == canonical_generation_method("ranking")
    assert canonical_generation_method("gcad_gss") == canonical_generation_method("both")


def test_valid_response_converts_to_native_pkl(tmp_path):
    backend, requests = export(tmp_path)
    write_jsonl(tmp_path / "generation_responses_raw.jsonl", response(requests))
    report = backend.validate(tmp_path)
    assert report["final_valid_sequence_count"] == 10
    path = backend.convert(
        tmp_path,
        day_mapping={"Monday": 0},
        hour_mapping={f"({i}~{i+3})": i // 3 for i in range(0, 24, 3)},
        device_mapping={"Light": 1},
        action_mapping={"Light:switch on": 10, "Light:switch off": 11},
    )
    assert len(pickle.loads(path.read_bytes())) == 10


def test_missing_and_extra_responses_fail(tmp_path):
    _, requests = export(tmp_path)
    records = response(requests)
    records[0]["request_id"] = "unknown"
    write_jsonl(tmp_path / "generation_responses_raw.jsonl", records)
    with pytest.raises(ValueError):
        validate_responses(tmp_path / "generation_requests.jsonl", tmp_path / "generation_responses_raw.jsonl", tmp_path)


def test_markdown_and_invalid_json_are_rejected(tmp_path):
    _, _ = export(tmp_path)
    (tmp_path / "generation_responses_raw.jsonl").write_text("```json\n")
    with pytest.raises(ValueError):
        validate_responses(tmp_path / "generation_requests.jsonl", tmp_path / "generation_responses_raw.jsonl", tmp_path)


def test_illegal_device_action_and_duplicates_are_detected(tmp_path):
    _, requests = export(tmp_path)
    records = response(requests, duplicate=True)
    records[0]["sequences"][0]["events"][0]["device"] = "Illegal"
    write_jsonl(tmp_path / "generation_responses_raw.jsonl", records)
    report = validate_responses(
        tmp_path / "generation_requests.jsonl", tmp_path / "generation_responses_raw.jsonl", tmp_path, raise_on_failure=False
    )
    assert report["illegal_device_count"] == 1
    assert report["duplicate_count"] > 0


def test_backend_has_no_api_key_or_network_dependency():
    backend = CodexFileBackend()
    assert backend.requires_api_key is False
    source = __import__("inspect").getsource(type(backend))
    assert "import requests" not in source and "import openai" not in source and "import socket" not in source
