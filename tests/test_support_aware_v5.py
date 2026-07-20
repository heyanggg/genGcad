import inspect
import json
from pathlib import Path

import pytest
import yaml

from SmartGen.generation_backends import support_aware_v5 as v5


ORIGINAL = {
    "0_0": 8, "0_1": 6, "1_0": 12, "1_1": 10, "2_0": 7, "2_1": 6,
    "3_0": 6, "3_1": 8, "4_0": 8, "4_1": 29, "4_2": 9, "5_0": 9,
    "5_1": 6, "6_0": 7, "6_1": 6,
}


def test_largest_remainder_candidate_quotas_are_exact_and_deterministic():
    expected = {
        "0_0": 9, "0_1": 7, "1_0": 14, "1_1": 12, "2_0": 8, "2_1": 7,
        "3_0": 7, "3_1": 9, "4_0": 9, "4_1": 34, "4_2": 11, "5_0": 11,
        "5_1": 7, "6_0": 8, "6_1": 7,
    }
    assert v5.largest_remainder_quotas(ORIGINAL, 160) == expected
    assert sum(expected.values()) == 160
    assert v5.largest_remainder_quotas(ORIGINAL, 160) == v5.largest_remainder_quotas(ORIGINAL, 160)


def _support_plan():
    return {
        "candidate_pool_sequence_support_target": 6,
        "selected_sequence_support_target": 5,
        "actions": {
            "Light:switch on": {
                "allowed_source_groups": ["0_0"],
                "candidate_pool_sequence_support_target": 6,
                "selected_sequence_support_target": 5,
            },
            "Television:setChannel": {
                "allowed_source_groups": ["0_0"],
                "candidate_pool_sequence_support_target": 6,
                "selected_sequence_support_target": 5,
            },
        },
        "contains_complete_event_sequences": False,
        "contains_per_sequence_actions": False,
        "contains_combination_formula": False,
        "programmatic_event_construction": False,
    }


def test_support_plan_has_no_events_or_per_sequence_action_assignments():
    plan = _support_plan()
    v5.validate_support_plan_v2(plan, set(plan["actions"]))
    assert "events" not in json.dumps(plan)
    poisoned = dict(plan)
    poisoned["per_sequence_actions"] = {"s1": ["Light:switch on"]}
    with pytest.raises(ValueError, match="event construction"):
        v5.validate_support_plan_v2(poisoned, set(plan["actions"]))


def test_all_actions_receive_the_same_targets_without_named_exceptions():
    plan = _support_plan()
    plan["actions"]["Light:switch on"]["candidate_pool_sequence_support_target"] = 5
    with pytest.raises(ValueError, match="non-uniform"):
        v5.validate_support_plan_v2(plan, set(plan["actions"]))
    source = inspect.getsource(v5)
    assert "AirPurifier:setAirPurifierMode" not in source
    assert "Television:setPictureMode" not in source


def test_candidate_sequence_support_counts_repeated_tokens_once():
    records = [
        {"actions": ["A:x", "A:x", "B:y"]},
        {"actions": ["A:x"]},
    ]
    assert v5.candidate_sequence_support(records) == {"A:x": 2, "B:y": 1}


def _records():
    return [
        {"sequence_id": "s1", "group_id": "g", "actions": ["A:x", "B:y"],
         "devices": ["A", "B"], "event_count": 3},
        {"sequence_id": "s2", "group_id": "g", "actions": ["A:x", "B:y", "A:x"],
         "devices": ["A", "B"], "event_count": 4},
        {"sequence_id": "s3", "group_id": "g", "actions": ["B:y", "A:x", "B:y", "A:x"],
         "devices": ["A", "B"], "event_count": 5},
    ]


def test_selection_constraints_refuse_action_support_or_group_quota_damage():
    plan = {"actions": {"A:x": {}, "B:y": {}}, "selected_sequence_support_target": 2,
            "groups": {"g": {"final_quota": 2}}}
    protocol = {"source_semantic_v2": {"maximum_target_metadata_only_token_share": 0.03,
                                        "minimum_source_transition_coverage": 0.0}}
    source = [["A:x", "B:y"], ["B:y", "A:x"]]
    transitions = {("A:x", "B:y"), ("B:y", "A:x")}
    good = v5._selection_constraints(_records(), plan=plan, protocol=protocol,
                                     source_sequences=source, source_transitions=transitions)
    assert good["passed"] is True
    broken = v5._selection_constraints(_records()[:1], plan=plan, protocol=protocol,
                                       source_sequences=source, source_transitions=transitions)
    assert broken["passed"] is False
    assert broken["checks"]["selected_action_support_target"] is False
    assert broken["checks"]["group_quotas_not_below_final"] is False


def test_v5_selector_source_has_no_event_constructor_or_supplemental_generation():
    source = inspect.getsource(v5.select_support_preserving_subset)
    assert "event[" not in source
    assert "supplemental_generation_used\": False" in source
    assert "target_normal" not in source and "target_attack" not in source


def test_v5_frozen_thresholds_match_v4_and_keep_gcad_ranking_off():
    v4 = yaml.safe_load(Path("configs/generation_protocol_v4/fr_spring.yaml").read_text())
    config = yaml.safe_load(Path("configs/generation_protocol_v5/fr_spring.yaml").read_text())
    assert config["source_semantic_v2"] == v4["source_semantic_v2"]
    assert config["split_feasibility_v1"] == v4["split_feasibility_v1"]
    assert config["reconstruction_health_v1"] == v4["reconstruction_health_v1"]
    assert config["gcad_source_enabled"] is False
    assert config["ranking_enabled"] is False
    assert config["uses_target_behavior"] is False


def test_replicate4_frozen_artifact_is_unchanged():
    import hashlib
    path = Path("outputs/codex_generation_v2/fr/spring/codex_gpt56_v4/replicate_4/checksums.sha256")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        "b0d5eaeafa62308a76af04c32f049d7eb39550aa9e9e95ce23f6b54863d50d16"
