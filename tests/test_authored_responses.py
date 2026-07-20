import json

from SmartGen.generation_backends.authored_responses import materialize_authored_responses


def test_authored_serializer_only_expands_explicit_actions_and_times(tmp_path):
    request = {
        "request_id": "r1", "experiment_id": "e", "group_id": "0_0", "generation_batch": 1,
        "requested_sequence_count": 1,
    }
    (tmp_path / "requests.jsonl").write_text(json.dumps(request) + "\n")
    plan = [{"request_id": "r1", "group_id": "0_0", "sequences": [{
        "sequence_id": "s1", "day": 0, "start_bin": 7,
        "actions": ["Light|switch on", "Blind|windowShade open"],
    }]}]
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    responses = materialize_authored_responses(tmp_path / "requests.jsonl", tmp_path / "plan.json", tmp_path / "raw.jsonl")
    events = responses[0]["sequences"][0]["events"]
    assert events[0] == {"day": "Monday", "hour": "(21~24)", "device": "Light", "action": "switch on"}
    assert events[1]["day"] == "Tuesday" and events[1]["hour"] == "(0~3)"
    assert responses[0]["generation_notes"]["serializer_is_non_generative"] is True
