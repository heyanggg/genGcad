import inspect
import json
import pickle
from pathlib import Path

import pytest

from SmartGen import dictionary
from SmartGen.generation_backends.codex56_provenance import (
    BACKEND,
    apply_generation_provenance_gate,
    validate_codex56_requests,
    validate_codex56_response_metadata,
)
from SmartGen.generation_backends.semantic_actions import (
    canonical_semantic_action,
    classify_action_spaces,
    static_legal_actions,
)
from SmartGen.generation_backends.source_semantic_v2 import evaluate_source_semantic_v2
from SmartGen.generation_backends.split_feasibility import build_coverage_constrained_split


def _request():
    return {
        "request_id": "r", "generation_backend": BACKEND,
        "model_family": "GPT-5.6 via Codex agent",
        "data_boundary": {"uses_target_behavior": False, "uses_target_normal": False,
                          "uses_target_attack": False, "uses_target_labels": False},
    }


def _response():
    return {
        "generation_backend": BACKEND, "content_author": "codex_gpt56_agent",
        "external_api_used": False, "api_key_used": False,
        "programmatic_event_construction": False, "target_behavior_used": False,
        "generation_notes": {"authored_plan_used": False, "python_events_constructed": False},
    }


def test_gpt56_backend_metadata_is_exact_and_does_not_fabricate_api_fields():
    validate_codex56_requests([_request()])
    validate_codex56_response_metadata(_response())
    assert not ({"api_request_id", "temperature", "token_usage", "sampling_seed"} & _response().keys())


@pytest.mark.parametrize("field", ["authored_plan", "event_template", "per_sequence_actions", "programmatic_events"])
def test_codex_mode_forbids_programmatic_request_fields(field):
    request = _request()
    request[field] = []
    with pytest.raises(ValueError, match="programmatic"):
        validate_codex56_requests([request])


def test_programmatic_output_cannot_be_marked_as_codex():
    response = _response()
    response["programmatic_event_construction"] = True
    with pytest.raises(ValueError, match="programmatic_event_construction"):
        validate_codex56_response_metadata(response)


def test_formal_codex_modules_do_not_call_authored_plan_or_construct_events():
    from SmartGen.generation_backends import codex56_pipeline, codex56_requests

    source = inspect.getsource(codex56_pipeline) + inspect.getsource(codex56_requests)
    assert "materialize_authored_responses" not in source
    assert "build_replicate3_authored_plan" not in source
    assert "sequence_id must use" not in source


def test_provenance_failure_blocks_before_later_artifacts(tmp_path):
    (tmp_path / "generation_requests.jsonl").write_text(json.dumps(_request()) + "\n")
    (tmp_path / "generation_responses_raw.jsonl").write_text("{}\n")
    (tmp_path / "agent_generation_attestation.json").write_text(json.dumps({
        "generation_backend": BACKEND, "content_author": "codex_gpt56_agent",
        "direct_agent_authorship": False, "external_api_used": False, "api_key_used": False,
        "programmatic_event_construction": False, "target_behavior_used": False,
    }))
    with pytest.raises(ValueError, match="provenance"):
        apply_generation_provenance_gate(tmp_path)
    assert not (tmp_path / "generation_quality_gate.json").exists()


def test_canonical_semantic_action_filter_is_shared_and_strict():
    metadata = {"Light": ["switch on"]}
    assert canonical_semantic_action("Light", "switch on", metadata) == "Light:switch on"
    for device, action in (("None", "location"), ("Other", "None:location"), ("", "switch on"),
                           ("Light", "unknown"), ("Missing", "switch on")):
        assert canonical_semantic_action(device, action, metadata) is None
    assert static_legal_actions(metadata) == {"Light:switch on"}


def _numeric(action):
    device, _ = action.split(":", 1)
    return [0, 0, dictionary.fr_devices_dict[device], dictionary.fr_actions[action]]


def test_source_observed_and_metadata_only_spaces_are_separate():
    metadata = {"Light": ["switch on", "switch off"]}
    spaces = classify_action_spaces([_numeric("Light:switch on")], "fr", metadata)
    assert spaces["source_observed_target_legal_actions"] == ["Light:switch on"]
    assert spaces["target_metadata_only_actions"] == ["Light:switch off"]


def _semantic_policy():
    return {
        "minimum_source_action_precision": 0.5, "minimum_global_source_vocabulary_recall": 1.0,
        "minimum_group_vocabulary_recall": 1.0, "minimum_frequency_weighted_recall": 1.0,
        "rare_source_sequence_support_maximum": 4, "minimum_rare_action_recall": 1.0,
        "minimum_sequence_support_per_action": 2, "maximum_low_support_action_ratio": 0.0,
        "maximum_target_metadata_only_token_share": 0.4, "minimum_source_transition_coverage": 0.5,
        "maximum_unseen_transition_ratio": 0.5, "minimum_source_transition_sequence_support": 1,
    }


def test_source_semantic_v2_reports_recall_static_usage_support_and_transitions():
    metadata = {"Light": ["switch on", "switch off"]}
    source = [_numeric("Light:switch on") + _numeric("Light:switch off")] * 2
    records = [
        {"sequence_id": "a", "group_id": "g", "actions": ["Light:switch on", "Light:switch off"],
         "devices": ["Light"], "event_count": 2},
        {"sequence_id": "b", "group_id": "g", "actions": ["Light:switch on", "Light:switch off"],
         "devices": ["Light"], "event_count": 2},
    ]
    report = evaluate_source_semantic_v2(
        records, source_numeric=source, group_source_numeric={"g": source}, dataset="fr",
        metadata=metadata, policy=_semantic_policy(),
    )
    assert report["metrics"]["global_source_vocabulary_recall"] == 1.0
    assert report["metrics"]["static_metadata_action_coverage"] == 1.0
    assert report["metrics"]["target_metadata_only_token_share"] == 0.0
    assert report["per_action_support"]["Light:switch on"]["sequence_support_count"] == 2
    assert report["metrics"]["source_transition_coverage"] == 1.0
    assert report["gate"]["passed"] is True


def _split_records(count=40):
    return [
        {"sequence_id": f"s{i}", "group_id": f"g{i % 4}",
         "actions": ["Light:switch on", "Light:switch off" if i % 3 else "Blind:windowShade open"],
         "devices": ["Light", "Blind"], "event_count": 3 + i % 4}
        for i in range(count)
    ]


def _split_policy():
    return {"train_validation_ratio": 0.8, "minimum_train_sequence_support_for_validation_action": 3,
            "minimum_group_size_for_validation": 5, "maximum_length_distribution_total_variation": 0.3}


def test_multilabel_split_is_reproducible_has_no_leakage_or_unseen_actions():
    first = build_coverage_constrained_split(_split_records(), _split_policy(), 2024)
    second = build_coverage_constrained_split(_split_records(), _split_policy(), 2024)
    assert first == second and first["passed"] is True
    assert set(first["train_sequence_ids"]).isdisjoint(first["validation_sequence_ids"])
    assert first["validation_unseen_action_count"] == 0
    assert first["validation_low_support_action_count"] == 0
    assert first["minimum_train_support_for_validation_actions"] >= 3


def test_all_fixed_splits_are_independently_required():
    manifests = [build_coverage_constrained_split(_split_records(), _split_policy(), seed)
                 for seed in (2024, 2025, 2026)]
    assert [item["seed"] for item in manifests] == [2024, 2025, 2026]
    assert all(item["passed"] for item in manifests)
    assert len({tuple(item["validation_sequence_ids"]) for item in manifests}) == 3


def test_infeasible_action_support_fails_instead_of_selecting_a_favorable_seed():
    records = _split_records(20)
    records[0]["actions"].append("Rare:only")
    policy = _split_policy()
    policy["minimum_train_sequence_support_for_validation_action"] = 100
    report = build_coverage_constrained_split(records, policy, 2024)
    assert report["passed"] is False


def test_reconstruction_health_v1_hash_and_replicate3_artifacts_remain_frozen():
    import hashlib
    assert hashlib.sha256(Path("configs/generation_protocol/reconstruction_health_v1.json").read_bytes()).hexdigest() \
        == "600c1ca86cc7e0b412521b1cc525ddd66fc3e21ff999f938a79ddc029a4a822c"
    assert hashlib.sha256(Path(
        "outputs/codex_generation_v2/fr/spring/baseline_source_copy_safe/replicate_3/checksums.sha256"
    ).read_bytes()).hexdigest() == "aee998c3d996bce9c0f533ba79ee7e1477ee5b7e2117766bac4f4b25c019e2ee"


def test_v4_protocol_keeps_gcad_ranking_and_target_behavior_off():
    import yaml
    config = yaml.safe_load(Path("configs/generation_protocol_v4/fr_spring.yaml").read_text())
    assert config["gcad_source_enabled"] is False and config["ranking_enabled"] is False
    assert config["uses_target_behavior"] is False
    assert config["reconstruction_health_v1"]["threshold_percentile"] == 95.5
