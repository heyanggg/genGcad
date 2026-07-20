import json

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
