import json
from pathlib import Path

import pytest

from SmartGen.generation_backends.source_copy_safe import (
    FROZEN_CONFIG_SHA256,
    action_sequence_fingerprint,
    load_frozen_config,
    sha256_file,
    validate_replacement_mapping,
)
from SmartGen.generation_backends.validation import validate_responses


CONFIG = Path("configs/generation_protocol/source_copy_safe_v1.json")


def _event(device="Light", action="switch on", day="Monday", hour="(0~3)"):
    return {"day": day, "hour": hour, "device": device, "action": action}


def test_source_copy_safe_config_is_frozen_and_recursively_immutable():
    config = load_frozen_config(CONFIG)
    assert sha256_file(CONFIG) == FROZEN_CONFIG_SHA256
    with pytest.raises(TypeError):
        config["version"] = "changed"
    with pytest.raises(TypeError):
        config["allowed_automatic_replacement_categories"][0] = "changed"


def test_action_fingerprint_ignores_day_and_hour():
    first = [_event(), _event(action="switch off", hour="(3~6)")]
    second = [_event(day="Sunday", hour="(18~21)"), _event(action="switch off", day="Sunday", hour="(21~24)")]
    assert action_sequence_fingerprint(first) == action_sequence_fingerprint(second)


def test_source_action_copy_is_hard_invalidity_even_with_new_time(tmp_path):
    events = [_event(day="Sunday", hour="(18~21)"), _event(action="switch off", day="Sunday", hour="(21~24)")]
    request = {
        "request_id": "r", "requested_sequence_count": 1,
        "target_static_device_metadata": {"Light": ["switch on", "switch off"]},
        "sequence_constraints": {"min_events": 2, "max_events": 3},
    }
    response = {
        "request_id": "r", "generation_backend": "codex_agent_file", "schema_version": "1.0",
        "sequences": [{"sequence_id": "s", "events": events}],
        "generation_notes": {"used_target_behavior": False, "used_target_labels": False,
                             "copied_from_existing_synthetic_data": False},
    }
    (tmp_path / "requests.jsonl").write_text(json.dumps(request) + "\n")
    (tmp_path / "responses.jsonl").write_text(json.dumps(response) + "\n")
    report = validate_responses(
        tmp_path / "requests.jsonl", tmp_path / "responses.jsonl", tmp_path,
        source_action_fingerprints={action_sequence_fingerprint(events)},
        action_only_duplicate_check=True, raise_on_failure=False,
    )
    assert report["source_copy_count"] == 1
    assert report["final_valid_sequence_count"] == 0


def test_replacements_are_limited_to_frozen_invalidity_categories():
    config = load_frozen_config(CONFIG)
    validate_replacement_mapping([{"category": "source_copy", "old": "a", "new": "b"}], config)
    with pytest.raises(ValueError, match="not allowed"):
        validate_replacement_mapping([{"category": "low_source_semantic_score"}], config)
    too_many = [{"category": "duplicate", "index": index} for index in range(21)]
    with pytest.raises(ValueError, match="exceeds"):
        validate_replacement_mapping(too_many, config)
