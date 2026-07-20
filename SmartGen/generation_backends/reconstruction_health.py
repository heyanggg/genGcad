from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "generation_protocol" / "reconstruction_health_v1.json"
CONFIG_SHA256 = "600c1ca86cc7e0b412521b1cc525ddd66fc3e21ff999f938a79ddc029a4a822c"


def load_frozen_policy(path: str | Path = CONFIG_PATH):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.resolve() == CONFIG_PATH.resolve() and digest != CONFIG_SHA256:
        raise ValueError(f"reconstruction_health_v1 config SHA256 mismatch: {digest}")
    return MappingProxyType(json.loads(path.read_text(encoding="utf-8")))


def _loss_mode_diagnostic(losses: np.ndarray, policy) -> dict:
    log_losses = np.sort(np.log10(np.maximum(losses, 1e-12)))
    if len(log_losses) < 2:
        return {"largest_log10_gap": 0.0, "split_index": None, "lower_cluster_fraction": 1.0,
                "upper_cluster_fraction": 0.0, "obvious_bimodality": False}
    gaps = np.diff(log_losses)
    split = int(np.argmax(gaps)) + 1
    lower_fraction = split / len(log_losses)
    upper_fraction = 1.0 - lower_fraction
    largest_gap = float(gaps[split - 1])
    obvious = (
        largest_gap >= policy["bimodality_log10_gap"]
        and min(lower_fraction, upper_fraction) >= policy["minimum_bimodal_cluster_fraction"]
    )
    return {
        "largest_log10_gap": largest_gap,
        "split_index": split,
        "lower_cluster_fraction": lower_fraction,
        "upper_cluster_fraction": upper_fraction,
        "obvious_bimodality": obvious,
    }


def evaluate_reconstruction_health(diagnostics: list[dict], policy=None) -> dict:
    policy = policy or load_frozen_policy()
    expected_seeds = list(policy["split_seeds"])
    by_seed = {int(item["split_integrity"]["seed"]): item for item in diagnostics}
    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError(f"reconstruction diagnostics must exactly cover split seeds {expected_seeds}")
    seed_reports = {}
    all_seed_checks = []
    thresholds = []
    for seed in expected_seeds:
        item = by_seed[seed]
        losses = np.asarray(item["validation_losses"], dtype=float)
        if not len(losses) or np.any(~np.isfinite(losses)) or np.any(losses < 0):
            raise ValueError(f"invalid validation losses for seed {seed}")
        top_count = max(1, int(math.ceil(policy["top_fraction"] * len(losses))))
        total_loss = float(losses.sum())
        top_contribution = float(np.sort(losses)[-top_count:].sum() / total_loss) if total_loss > 0 else 1.0
        quantiles = {
            "p10": float(np.percentile(losses, 10)),
            "p25": float(np.percentile(losses, 25)),
            "p50": float(np.percentile(losses, 50)),
            "p75": float(np.percentile(losses, 75)),
            "p90": float(np.percentile(losses, 90)),
            "p95_5": float(np.percentile(losses, 95.5)),
            "p99": float(np.percentile(losses, 99)),
        }
        mode = _loss_mode_diagnostic(losses, policy)
        metrics = {
            "validation_sequence_count": len(losses),
            "near_zero_loss_ratio": float(np.mean(losses <= policy["near_zero_loss_cutoff"])),
            "quantiles": quantiles,
            "p90_minus_p10": quantiles["p90"] - quantiles["p10"],
            "top_fraction_count": top_count,
            "top_fraction_loss_contribution": top_contribution,
            "loss_mode_diagnostic": mode,
            "threshold": float(item["threshold"]),
            "train_validation_exact_overlap": int(item["split_integrity"]["exact_overlap_count"]),
        }
        checks = {
            "validation_count": len(losses) >= policy["minimum_validation_sequence_count"],
            "exact_overlap_zero": metrics["train_validation_exact_overlap"]
            == policy["required_train_validation_exact_overlap"],
            "near_zero_ratio": metrics["near_zero_loss_ratio"] <= policy["maximum_near_zero_loss_ratio"],
            "loss_dispersion": metrics["p90_minus_p10"] >= policy["minimum_p90_minus_p10"],
            "top_loss_not_dominant": top_contribution <= policy["maximum_top_fraction_loss_contribution"],
            "not_obviously_bimodal": not mode["obvious_bimodality"],
        }
        seed_reports[str(seed)] = {"metrics": metrics, "checks": checks, "passed": all(checks.values())}
        all_seed_checks.append(all(checks.values()))
        thresholds.append(float(item["threshold"]))
    threshold_array = np.asarray(thresholds, dtype=float)
    threshold_mean = float(threshold_array.mean())
    threshold_cv = float(threshold_array.std() / threshold_mean) if threshold_mean > 0 else float("inf")
    threshold_relative_range = (
        float((threshold_array.max() - threshold_array.min()) / np.median(threshold_array))
        if float(np.median(threshold_array)) > 0 else float("inf")
    )
    stability = {
        "split_seeds": expected_seeds,
        "thresholds": thresholds,
        "coefficient_of_variation": threshold_cv,
        "relative_range": threshold_relative_range,
        "checks": {
            "threshold_cv": threshold_cv <= policy["maximum_threshold_coefficient_of_variation"],
            "threshold_relative_range": threshold_relative_range <= policy["maximum_threshold_relative_range"],
        },
    }
    checks = {
        "all_split_health_checks": all(all_seed_checks),
        "threshold_stability": all(stability["checks"].values()),
    }
    return {
        "policy": dict(policy),
        "policy_sha256": CONFIG_SHA256,
        "seed_reports": seed_reports,
        "threshold_stability": stability,
        "gate": {
            "stage": "pre_target_reconstruction_health_v1",
            "passed": all(checks.values()),
            "checks": checks,
            "threshold_basis": "generated-only reconstruction losses across frozen split seeds",
            "uses_official_synthetic_threshold": False,
            "uses_target_behavior": False,
        },
    }


def apply_reconstruction_health_gate(directory: str | Path, diagnostics_paths: list[str | Path]) -> dict:
    directory = Path(directory)
    diagnostics = [json.loads(Path(path).read_text(encoding="utf-8")) for path in diagnostics_paths]
    result = evaluate_reconstruction_health(diagnostics)
    (directory / "reconstruction_health_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (directory / "reconstruction_health_gate.json").write_text(json.dumps(result["gate"], indent=2), encoding="utf-8")
    return result


def require_final_evaluation_gates(prepared_dir: str | Path) -> None:
    experiment = Path(prepared_dir).parent
    required = (
        "generation_quality_gate.json",
        "source_semantic_gate.json",
        "reconstruction_health_gate.json",
    )
    for name in required:
        path = experiment / name
        if not path.exists():
            raise PermissionError(f"final evaluation blocked; gate missing: {name}")
        gate = json.loads(path.read_text(encoding="utf-8"))
        if gate.get("uses_target_behavior") is not False or gate.get("passed") is not True:
            raise PermissionError(f"final evaluation blocked; gate not passed with zero-target declaration: {name}")
