import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from SmartGen.generation_backends.frozen_protocol import (
    REPLICATE_2_REQUESTS_SHA256,
    verify_frozen_requests,
)
from SmartGen.generation_backends.reconstruction_health import (
    CONFIG_SHA256,
    evaluate_reconstruction_health,
    load_frozen_policy,
    require_final_evaluation_gates,
)
from SmartGen.generation_backends.source_semantic_gate import POLICY
from SmartGen.gcad_source.data_boundary import DataBoundaryError, require_roles
from SmartGen.gcad_source.data_roles import DataRole, RoleBoundPath


def _diagnostic(seed, losses, threshold=1.0, overlap=0):
    return {
        "validation_losses": list(map(float, losses)),
        "threshold": float(threshold),
        "split_integrity": {"seed": seed, "exact_overlap_count": overlap},
        "uses_target_behavior": False,
    }


def _three(losses, thresholds=(1.0, 1.05, 0.95)):
    return [_diagnostic(seed, losses, threshold) for seed, threshold in zip((2024, 2025, 2026), thresholds)]


def test_replicate_2_requests_hash_is_frozen_and_mismatch_is_rejected(tmp_path):
    assert REPLICATE_2_REQUESTS_SHA256 == "7c0fd1766598b4cfacdbcfac31fc1f672416dbc80c652ed4ddd8f4d9439c6860"
    actual = Path("outputs/codex_generation_v2/fr/spring/baseline_source_semantic/replicate_2/generation_requests.jsonl")
    if actual.exists():
        assert verify_frozen_requests(actual) == REPLICATE_2_REQUESTS_SHA256
    candidate = tmp_path / "generation_requests.jsonl"
    candidate.write_text("changed\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_frozen_requests(candidate)


def test_source_semantic_v1_policy_is_immutable_and_has_frozen_values():
    canonical = json.dumps(dict(POLICY), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == "62ad8a523bb01f2a8dea44b458e2970274dcfa8c713fbd3426e397227f3d15a2"
    with pytest.raises(TypeError):
        POLICY["minimum_group_anchor_share"] = 0.0


def test_reconstruction_health_policy_is_frozen_before_losses_are_read():
    policy = load_frozen_policy()
    assert CONFIG_SHA256 == "600c1ca86cc7e0b412521b1cc525ddd66fc3e21ff999f938a79ddc029a4a822c"
    assert policy["uses_target_behavior"] is False
    assert policy["uses_official_synthetic_threshold"] is False
    with pytest.raises(TypeError):
        policy["maximum_near_zero_loss_ratio"] = 1.0


def test_near_zero_validation_collapse_is_rejected():
    report = evaluate_reconstruction_health(_three(np.full(30, 1e-10)))
    seed = report["seed_reports"]["2024"]
    assert seed["metrics"]["near_zero_loss_ratio"] == 1.0
    assert seed["checks"]["near_zero_ratio"] is False
    assert report["gate"]["passed"] is False


def test_small_high_loss_tail_dominating_threshold_is_rejected():
    losses = np.asarray([0.1] * 27 + [100.0] * 3)
    report = evaluate_reconstruction_health(_three(losses, (90.0, 92.0, 88.0)))
    seed = report["seed_reports"]["2024"]
    assert seed["metrics"]["top_fraction_loss_contribution"] > 0.5
    assert seed["checks"]["top_loss_not_dominant"] is False


def test_obvious_bimodal_validation_losses_are_rejected():
    losses = np.asarray([0.001] * 15 + [1.0] * 15)
    report = evaluate_reconstruction_health(_three(losses))
    mode = report["seed_reports"]["2024"]["metrics"]["loss_mode_diagnostic"]
    assert mode["obvious_bimodality"] is True
    assert report["seed_reports"]["2024"]["checks"]["not_obviously_bimodal"] is False


def test_multi_split_threshold_stability_is_calculated_correctly():
    losses = np.linspace(0.1, 1.0, 30)
    stable = evaluate_reconstruction_health(_three(losses, (1.0, 1.05, 0.95)))
    metrics = stable["threshold_stability"]
    assert np.isclose(metrics["coefficient_of_variation"], np.std([1.0, 1.05, 0.95]))
    assert np.isclose(metrics["relative_range"], 0.1)
    assert stable["gate"]["passed"] is True
    unstable = evaluate_reconstruction_health(_three(losses, (1.0, 3.0, 5.0)))
    assert unstable["gate"]["checks"]["threshold_stability"] is False


def test_tof_and_reconstruction_health_refuse_target_roles():
    target = RoleBoundPath.build("target.pkl", DataRole.TARGET_NORMAL_EVAL)
    with pytest.raises(DataBoundaryError):
        require_roles("tof", [target])
    with pytest.raises(DataBoundaryError):
        require_roles("threshold", [target])
    report = evaluate_reconstruction_health(_three(np.linspace(0.1, 1.0, 30)))
    assert report["gate"]["uses_target_behavior"] is False


def test_final_evaluation_is_blocked_until_all_three_gates_pass(tmp_path):
    prepared = tmp_path / "experiment" / "downstream_prepared"
    prepared.mkdir(parents=True)
    with pytest.raises(PermissionError, match="missing"):
        require_final_evaluation_gates(prepared)
    for name in ("generation_quality_gate.json", "source_semantic_gate.json", "reconstruction_health_gate.json"):
        (prepared.parent / name).write_text(json.dumps({"passed": True, "uses_target_behavior": False}))
    (prepared.parent / "source_semantic_gate.json").write_text(
        json.dumps({"passed": False, "uses_target_behavior": False})
    )
    with pytest.raises(PermissionError, match="not passed"):
        require_final_evaluation_gates(prepared)
    (prepared.parent / "source_semantic_gate.json").write_text(
        json.dumps({"passed": True, "uses_target_behavior": False})
    )
    require_final_evaluation_gates(prepared)


def test_codex_v2_prepared_evaluation_invokes_three_gate_guard(tmp_path, monkeypatch):
    from SmartGen.gcad_source import downstream_evaluation as module

    prepared = tmp_path / "outputs" / "codex_generation_v2" / "replicate_3" / "downstream_prepared"
    prepared.mkdir(parents=True)
    called = []
    monkeypatch.setattr(
        "SmartGen.generation_backends.reconstruction_health.require_final_evaluation_gates",
        lambda path: called.append(Path(path)),
    )
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        module.evaluate_prepared_detector(prepared, "fr", "spring")
    assert called == [prepared]
