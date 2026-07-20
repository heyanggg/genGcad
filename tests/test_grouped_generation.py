import json
import pickle

import pytest

from SmartGen import dictionary
from SmartGen.generation_backends.grouped_requests import (
    derive_source_length_summary,
    export_grouped_baseline_requests,
)
from SmartGen.generation_backends.validation import validate_responses


def setup_groups(tmp_path):
    source = tmp_path / "source/fr/winter"
    source.mkdir(parents=True)
    device = dictionary.fr_devices_dict["Light"]
    on = dictionary.fr_actions["Light:switch on"]
    off = dictionary.fr_actions["Light:switch off"]
    groups = {
        "0_0": [[0, 0, device, on], [0, 0, device, on, 0, 1, device, off]],
        "1_0": [[1, 0, device, on, 1, 1, device, off, 1, 2, device, on]],
    }
    for group, sequences in groups.items():
        (source / f"trn_day_{group}_SPPC_th=0.918.pkl").write_bytes(pickle.dumps(sequences))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"groups": [
        {"group_id": "0_0", "source_date_or_partition": "day_0", "requested_sequence_count": 2},
        {"group_id": "1_0", "source_date_or_partition": "day_1", "requested_sequence_count": 2},
    ]}))
    metadata = tmp_path / "metadata.json"; metadata.write_text('{"Light":["switch on","switch off"]}')
    gss = tmp_path / "gss.json"; gss.write_text("{}")
    control = tmp_path / "control.txt"; control.write_text("Light: switch on, switch off")
    return source.parent.parent, plan, metadata, gss, control


def test_grouped_requests_are_traceable_and_prompts_do_not_merge_groups(tmp_path):
    source_root, plan, metadata, gss, control = setup_groups(tmp_path)
    output = tmp_path / "outputs/codex_generation_v2/fr/spring/baseline/replicate_1"
    requests = export_grouped_baseline_requests(
        output, experiment_id="a2", dataset="fr", source_context="winter", target_context="spring",
        compression_threshold=0.918, group_plan_path=plan, target_metadata_path=metadata,
        original_gss_path=gss, device_control_path=control, source_root=source_root,
    )
    assert {item["group_id"] for item in requests} == {"0_0", "1_0"}
    assert {item["group_id"]: item["representative_sequence_count"] for item in requests} == {"0_0": 2, "1_0": 1}
    assert all(item["uses_target_behavior"] is False for item in requests)
    assert requests[0]["representative_sequence_ids"] != requests[1]["representative_sequence_ids"]
    assert "source group 0_0" in requests[0]["prompt"]
    assert "source group 1_0" not in requests[0]["prompt"]
    assert requests[0]["sequence_constraints"]["min_events"] == 2
    assert requests[0]["source_length_summary"]["derivation"].startswith("max(2")


def test_grouped_output_cannot_overwrite_a1(tmp_path):
    source_root, plan, metadata, gss, control = setup_groups(tmp_path)
    with pytest.raises(ValueError, match="codex_generation_v2"):
        export_grouped_baseline_requests(
            tmp_path / "outputs/codex_generation/fr/spring/baseline", experiment_id="a2", dataset="fr",
            source_context="winter", target_context="spring", compression_threshold=0.918,
            group_plan_path=plan, target_metadata_path=metadata, original_gss_path=gss,
            device_control_path=control, source_root=source_root,
        )


def test_length_bounds_are_source_derived_and_not_target_based():
    summary = derive_source_length_summary([[0, 0, 1, 1], [0, 0, 1, 1] * 6])
    assert summary["min"] == 1 and summary["max"] == 6
    assert summary["allowed_min"] >= 2
    assert "target" not in summary["derivation"]


def test_cross_group_duplicate_is_rejected(tmp_path):
    source_root, plan, metadata, gss, control = setup_groups(tmp_path)
    output = tmp_path / "outputs/codex_generation_v2/fr/spring/baseline/replicate_1"
    requests = export_grouped_baseline_requests(
        output, experiment_id="a2", dataset="fr", source_context="winter", target_context="spring",
        compression_threshold=0.918, group_plan_path=plan, target_metadata_path=metadata,
        original_gss_path=gss, device_control_path=control, source_root=source_root,
    )
    records = []
    for request in requests:
        sequence = {"sequence_id": f"s_{request['group_id']}", "events": [
            {"day": "Monday", "hour": "(0~3)", "device": "Light", "action": "switch on"},
            {"day": "Monday", "hour": "(3~6)", "device": "Light", "action": "switch off"},
        ]}
        records.append({
            "request_id": request["request_id"], "experiment_id": "a2", "method": "baseline_v2",
            "generation_backend": "codex_agent_file", "generation_batch": request["generation_batch"],
            "sequences": [dict(sequence, sequence_id=f"{sequence['sequence_id']}_{i}") for i in range(2)],
            "generation_notes": {"used_target_behavior": False, "used_target_labels": False,
                                 "copied_from_existing_synthetic_data": False}, "schema_version": "1.0",
        })
    output.joinpath("generation_responses_raw.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    report = validate_responses(output / "generation_requests.jsonl", output / "generation_responses_raw.jsonl", output, raise_on_failure=False)
    assert report["duplicate_count"] >= 1
