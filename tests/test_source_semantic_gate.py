import json
import pickle

import pytest

from SmartGen.generation_backends.source_semantic_gate import (
    _assert_source_only_provenance,
    require_pre_tof_gates,
    semantic_support_report,
)


def _source_fixture():
    sequences = [
        ["Light:switch on", "Blind:windowShade open"],
        ["Light:switch on", "Blind:windowShade close"],
        ["Light:switch on", "Blind:windowShade open"],
        ["Light:switch on", "Blind:windowShade close"],
    ]
    return sequences, [0, 0, 1, 1]


def test_source_anchored_generation_passes_source_only_semantic_gate():
    source, days = _source_fixture()
    generated = {"0_0": [source[0], source[1]]}
    report = semantic_support_report(generated, source, days, {"0_0": source[:2]})
    assert report["gate"]["passed"] is True
    assert report["metrics"]["mean_global_action_coverage"] == 1.0
    assert report["metrics"]["group_anchor_share"] == 1.0
    assert report["gate"]["uses_target_behavior"] is False
    assert report["thresholds"]["minimum_mean_global_action_coverage"] == (
        0.5 * report["calibration"]["mean_action_coverage"]
    )


def test_legal_but_source_detached_generation_fails_all_semantic_anchors():
    source, days = _source_fixture()
    generated = {"0_0": [["Oven:switch on", "Washer:act"], ["Camera:switch on", "Siren:alarm both"]]}
    report = semantic_support_report(generated, source, days, {"0_0": source[:2]})
    assert report["gate"]["passed"] is False
    assert report["metrics"]["zero_global_anchor_share"] == 1.0
    assert report["gate"]["checks"]["global_action_coverage"] is False
    assert report["gate"]["checks"]["group_anchor_share"] is False


def test_source_provenance_guard_rejects_target_declaration_and_mixed_context(tmp_path):
    source = tmp_path / "winter" / "split_trn.pkl"
    good = [{"uses_target_behavior": False, "source_group_path": str(source.parent / "group.pkl")}]
    _assert_source_only_provenance(good, source)
    with pytest.raises(ValueError, match="zero-target"):
        _assert_source_only_provenance([{**good[0], "uses_target_behavior": True}], source)
    with pytest.raises(ValueError, match="same source-context"):
        _assert_source_only_provenance(
            [{"uses_target_behavior": False, "source_group_path": str(tmp_path / "spring" / "group.pkl")}],
            source,
        )


def test_v2_tof_requires_both_passing_zero_target_gates(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        require_pre_tof_gates(tmp_path)
    (tmp_path / "generation_quality_gate.json").write_text(json.dumps({
        "passed": True, "uses_target_behavior": False,
    }))
    (tmp_path / "source_semantic_gate.json").write_text(json.dumps({
        "passed": False, "uses_target_behavior": False,
    }))
    with pytest.raises(ValueError, match="failed"):
        require_pre_tof_gates(tmp_path)
    (tmp_path / "source_semantic_gate.json").write_text(json.dumps({
        "passed": True, "uses_target_behavior": False,
    }))
    require_pre_tof_gates(tmp_path)


def test_diagnostic_only_report_cannot_create_or_replace_formal_gate(tmp_path, monkeypatch):
    from SmartGen.generation_backends import source_semantic_gate as module

    monkeypatch.setattr(module, "load_jsonl", lambda path: [])
    with pytest.raises(ValueError, match="failed_upstream"):
        module.diagnose_source_semantics(tmp_path, "fr", tmp_path / "source.pkl", diagnostic_only=True)


def test_diagnostic_only_writes_only_diagnostic_and_keeps_formal_status_unchanged(tmp_path):
    from SmartGen import dictionary
    from SmartGen.generation_backends import source_semantic_gate as module

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    device = dictionary.fr_devices_dict["Light"]
    action = dictionary.fr_actions["Light:switch on"]
    numeric = [[0, 0, device, action], [1, 0, device, action]]
    source = source_dir / "split_trn.pkl"; source.write_bytes(pickle.dumps(numeric))
    group = source_dir / "group.pkl"; group.write_bytes(pickle.dumps(numeric))
    request = {"request_id": "r", "group_id": "g", "source_group_path": str(group),
               "uses_target_behavior": False, "target_static_device_metadata": {"Light": ["switch on"]}}
    response = {"request_id": "r", "sequences": [{"sequence_id": "s", "events": [
        {"day": "Monday", "hour": "(0~3)", "device": "Light", "action": "switch on"}
    ]}]}
    (tmp_path / "generation_requests.jsonl").write_text(json.dumps(request) + "\n")
    (tmp_path / "generation_responses_validated.jsonl").write_text(json.dumps(response) + "\n")
    result = module.diagnose_source_semantics(
        tmp_path, "fr", source, diagnostic_only=True, formal_gate_status="failed_upstream"
    )
    assert result["diagnostic_only"] is True and result["formal_gate_status"] == "unchanged"
    assert result["upstream_formal_gate_status"] == "failed_upstream"
    assert (tmp_path / "source_semantic_diagnostic_only.json").exists()
    assert not (tmp_path / "source_semantic_gate.json").exists()
