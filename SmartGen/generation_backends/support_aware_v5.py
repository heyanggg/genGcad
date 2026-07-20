from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import yaml

from SmartGen.gcad_source.cell_builder import numeric_to_text
from SmartGen.gcad_source.data_boundary import boundary_declaration
from SmartGen.gcad_source.prompt_adapter import build_original_smartgen_prompt

from .codex56_provenance import AUTHOR, BACKEND, sha256_file, validate_codex56_requests
from .generation_diagnostics import _sequence_similarity, sequence_metrics
from .grouped_requests import _balanced_chunks, _context_sentence, derive_source_length_summary
from .response_loader import load_jsonl
from .schemas import SCHEMA_VERSION, prompt_sha256
from .semantic_actions import classify_action_spaces, event_semantic_actions, numeric_semantic_actions
from .validation import validate_responses


FORBIDDEN_SUPPORT_PLAN_KEYS = {
    "events", "event_sequence", "complete_sequence", "per_sequence_actions",
    "authored_plan", "event_template", "rotation", "modulo", "cartesian_product",
}


def largest_remainder_quotas(original: dict[str, int], total: int) -> dict[str, int]:
    if not original or total < 0 or any(value < 0 for value in original.values()):
        raise ValueError("quota inputs must be non-negative and non-empty")
    denominator = sum(original.values())
    if denominator <= 0:
        raise ValueError("original quota sum must be positive")
    raw = {key: Fraction(total * value, denominator) for key, value in original.items()}
    result = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(original, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    if sum(result.values()) != total:
        raise AssertionError("largest-remainder total mismatch")
    return dict(sorted(result.items()))


def _length_quota(count: int, minimum: int, maximum: int) -> dict[str, int]:
    return largest_remainder_quotas({str(length): 1 for length in range(minimum, maximum + 1)}, count)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_support_plan_v2(plan: dict, expected_actions: set[str] | None = None) -> None:
    keys = {key.lower() for key in _walk_keys(plan)}
    if keys & FORBIDDEN_SUPPORT_PLAN_KEYS:
        raise ValueError("support plan contains event construction or per-sequence content")
    if any(plan.get(flag) is not False for flag in (
        "contains_complete_event_sequences", "contains_per_sequence_actions",
        "contains_combination_formula", "programmatic_event_construction",
    )):
        raise ValueError("support plan boundary declaration failed")
    actions = plan.get("actions", {})
    if expected_actions is not None and set(actions) != expected_actions:
        raise ValueError("support targets must apply uniformly to every source action")
    for action, record in actions.items():
        if record.get("candidate_pool_sequence_support_target") != plan["candidate_pool_sequence_support_target"]:
            raise ValueError(f"non-uniform candidate support target: {action}")
        if record.get("selected_sequence_support_target") != plan["selected_sequence_support_target"]:
            raise ValueError(f"non-uniform selected support target: {action}")
        if not record.get("allowed_source_groups"):
            raise ValueError(f"source action lacks an evidence-backed group: {action}")


def _source_day_groups(source_full: list[list[int]], dataset: str, metadata: dict, group_ids: list[str]) -> dict[str, set[str]]:
    day_actions: dict[str, set[str]] = defaultdict(set)
    for sequence in source_full:
        if not sequence:
            continue
        day = str(int(sequence[0]))
        day_actions[day].update(numeric_semantic_actions(sequence, dataset, metadata))
    result: dict[str, set[str]] = defaultdict(set)
    for group_id in group_ids:
        day = group_id.split("_", 1)[0]
        for action in day_actions.get(day, set()):
            result[action].add(group_id)
    return result


def export_support_aware_v5_requests(
    output_dir: str | Path,
    *,
    preregistration_dir: str | Path,
    experiment_id: str,
    dataset: str,
    source_context: str,
    target_context: str,
    compression_threshold: float,
    group_plan_path: str | Path,
    protocol_path: str | Path,
    source_full_path: str | Path,
    target_metadata_path: str | Path,
    original_gss_path: str | Path,
    device_control_path: str | Path,
    source_root: str | Path = "SmartGen/IoT_data",
) -> list[dict]:
    output = Path(output_dir)
    prereg = Path(preregistration_dir)
    output.mkdir(parents=True, exist_ok=True)
    prereg.mkdir(parents=True, exist_ok=True)
    archive = output / "prompt_archive"
    archive.mkdir(exist_ok=True)
    protocol_text = Path(protocol_path).read_text(encoding="utf-8")
    protocol = yaml.safe_load(protocol_text)
    if protocol.get("version") != "generation_protocol_v5":
        raise ValueError("not a generation protocol v5 configuration")
    if protocol["generation_backend"] != BACKEND or protocol["programmatic_event_construction"] is not False:
        raise ValueError("invalid formal Codex v5 provenance configuration")
    plan = json.loads(Path(group_plan_path).read_text(encoding="utf-8"))
    original = {str(item["group_id"]): int(item["requested_sequence_count"]) for item in plan["groups"]}
    candidate_quotas = largest_remainder_quotas(original, int(protocol["candidate_pool_size"]))
    quota_artifact = {
        "algorithm": "largest_remainder",
        "formula": "candidate_pool_size * original_group_quota / final_sequence_count; floor then descending remainder; group_id ascending tie-break",
        "candidate_pool_size": protocol["candidate_pool_size"],
        "final_sequence_count": protocol["final_sequence_count"],
        "groups": {
            group: {"original_final_quota": original[group], "candidate_quota": candidate_quotas[group]}
            for group in sorted(original)
        },
        "uses_generation_results": False,
        "uses_target_behavior": False,
    }
    quota_path = prereg / "group_candidate_quotas.json"
    quota_path.write_text(json.dumps(quota_artifact, indent=2) + "\n", encoding="utf-8")

    metadata = json.loads(Path(target_metadata_path).read_text(encoding="utf-8"))
    gss = json.loads(Path(original_gss_path).read_text(encoding="utf-8"))
    device_control = Path(device_control_path).read_text(encoding="utf-8")
    source_full = pickle.loads(Path(source_full_path).read_bytes())
    spaces = classify_action_spaces(source_full, dataset, metadata)
    source_actions = set(spaces["source_observed_target_legal_actions"])
    group_cache = {}
    direct_groups: dict[str, set[str]] = defaultdict(set)
    for group_id in sorted(original):
        source_path = Path(source_root) / dataset / source_context / f"trn_day_{group_id}_SPPC_th={compression_threshold}.pkl"
        numeric = pickle.loads(source_path.read_bytes())
        sequences = [numeric_semantic_actions(sequence, dataset, metadata) for sequence in numeric]
        actions = sorted({action for sequence in sequences for action in sequence})
        for action in actions:
            direct_groups[action].add(group_id)
        transitions = sorted({pair for sequence in sequences for pair in zip(sequence, sequence[1:])})
        summary = derive_source_length_summary(numeric)
        minimum = max(protocol["length_policy"]["minimum_events"], summary["allowed_min"])
        maximum = min(protocol["length_policy"]["maximum_events"], summary["allowed_max"])
        if minimum > maximum:
            maximum = minimum
        group_cache[group_id] = {
            "path": source_path, "numeric": numeric, "representatives": numeric_to_text(numeric, dataset),
            "actions": actions, "transitions": [list(pair) for pair in transitions],
            "minimum": minimum, "maximum": maximum,
        }
    day_groups = _source_day_groups(source_full, dataset, metadata, sorted(original))
    action_records = {}
    group_targets: dict[str, dict[str, int]] = defaultdict(dict)
    for action in sorted(source_actions):
        allowed = sorted(direct_groups.get(action) or day_groups.get(action, set()))
        if not allowed:
            raise ValueError(f"cannot place source action in an evidence-backed group: {action}")
        weights = {group: candidate_quotas[group] for group in allowed}
        allocation = largest_remainder_quotas(weights, protocol["support_targets"]["candidate_pool_per_action"])
        for group, count in allocation.items():
            if count:
                group_targets[group][action] = count
        action_records[action] = {
            "allowed_source_groups": allowed,
            "group_evidence": "direct_sppc" if direct_groups.get(action) else "source_normal_day",
            "candidate_pool_sequence_support_target": protocol["support_targets"]["candidate_pool_per_action"],
            "selected_sequence_support_target": protocol["support_targets"]["selected_per_action"],
            "formal_source_semantic_v2_minimum": protocol["source_semantic_v2"]["minimum_sequence_support_per_action"],
            "candidate_group_support_allocation": allocation,
        }
    support_plan = {
        "version": "source_support_plan_v2",
        "role": "group-level support guidance for direct Codex authorship; never an event plan",
        "candidate_pool_sequence_support_target": protocol["support_targets"]["candidate_pool_per_action"],
        "selected_sequence_support_target": protocol["support_targets"]["selected_per_action"],
        "formal_source_semantic_v2_minimum": protocol["source_semantic_v2"]["minimum_sequence_support_per_action"],
        "source_observed_target_legal_action_count": len(source_actions),
        "actions": action_records,
        "groups": {
            group: {
                "candidate_quota": candidate_quotas[group], "final_quota": original[group],
                "candidate_action_sequence_support_targets": dict(sorted(group_targets[group].items())),
                "source_transition_anchors": group_cache[group]["transitions"],
                "length_distribution_target": _length_quota(candidate_quotas[group], group_cache[group]["minimum"], group_cache[group]["maximum"]),
            }
            for group in sorted(original)
        },
        "maximum_target_metadata_only_token_share": protocol["source_semantic_v2"]["maximum_target_metadata_only_token_share"],
        "contains_complete_event_sequences": False,
        "contains_per_sequence_actions": False,
        "contains_combination_formula": False,
        "programmatic_event_construction": False,
        "uses_target_behavior": False,
    }
    validate_support_plan_v2(support_plan, source_actions)
    support_path = prereg / "source_support_plan_v2.json"
    support_path.write_text(json.dumps(support_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prereg.joinpath("preregistration.yaml").write_text(protocol_text, encoding="utf-8")
    prereg.joinpath("preregistration.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    selector_path = prereg / "selector_config.json"
    selector_path.write_text(json.dumps(protocol["selector"], indent=2) + "\n", encoding="utf-8")
    (output / "selector_config.json").write_bytes(selector_path.read_bytes())

    requests = []
    for group_id in sorted(original):
        cached = group_cache[group_id]
        base_prompt = build_original_smartgen_prompt(
            device_control, _context_sentence(source_context, target_context), cached["representatives"], gss,
        )
        chunks = _balanced_chunks(candidate_quotas[group_id], 20)
        action_targets = support_plan["groups"][group_id]["candidate_action_sequence_support_targets"]
        per_batch_targets = {
            action: largest_remainder_quotas({str(index + 1): count for index, count in enumerate(chunks)}, target)
            for action, target in action_targets.items()
        }
        for group_batch, count in enumerate(chunks, 1):
            batch_targets = {
                action: allocation[str(group_batch)] for action, allocation in per_batch_targets.items()
                if allocation[str(group_batch)] > 0
            }
            guidance = {
                "group_id": group_id,
                "group_candidate_quota": candidate_quotas[group_id],
                "current_batch_candidate_count": count,
                "group_source_actions": cached["actions"],
                "group_candidate_action_sequence_support_targets": action_targets,
                "current_batch_action_sequence_support_targets": batch_targets,
                "source_transition_anchors": cached["transitions"],
                "length_distribution_target": support_plan["groups"][group_id]["length_distribution_target"],
            }
            prompt = base_prompt + (
                "\n\nFormal A5 support-aware Codex GPT-5.6 candidate-pool instructions:\n"
                f"Author exactly {count} independent candidates for source SPPC group {group_id}; each must have "
                f"{cached['minimum']} to {cached['maximum']} events. The active Codex agent must create every event "
                "directly. Python, formulas, loops, rotations, templates, Cartesian products, and GSS constructors must "
                "not create or suggest event content. Treat the support numbers as whole-batch goals, never as a "
                "per-sequence action assignment. Spread each requested action across distinct candidates; repeating an "
                "action inside one candidate counts only once. Keep every sequence independently plausible in warm "
                "spring, preserve natural source transitions, and do not force unrelated actions together. Do not copy "
                "a representative action-for-action, mechanically recite the plan, or use a fixed opening/ending. Use "
                "only static-legal device/action pairs and predominantly source-observed actions. The complete frozen "
                "pool has 160 candidates and will be deterministically reduced to 137 without changing events. Return "
                "one JSON response record matching the frozen schema.\n"
                f"Group-level support guidance (not an event plan): {json.dumps(guidance, ensure_ascii=False)}"
            )
            prompt_hash = prompt_sha256(prompt)
            request_id = "req5_" + hashlib.sha256(
                f"{experiment_id}|{group_id}|{group_batch}|5|{prompt_hash}".encode()
            ).hexdigest()[:20]
            request = {
                "request_id": request_id, "experiment_id": experiment_id, "dataset": dataset,
                "group_id": group_id, "group_batch": group_batch, "replicate": 5,
                "generation_backend": BACKEND, "model_family": "GPT-5.6 via Codex agent",
                "requested_sequence_count": count, "candidate_sequence_count": count,
                "final_group_sequence_count": original[group_id],
                "representative_sequences": cached["representatives"],
                "source_action_anchors": cached["actions"], "source_transition_hints": cached["transitions"],
                "group_support_targets": guidance,
                "target_context_description": {"change": f"{source_context} to {target_context}", "static_only": True},
                "target_static_device_action_mapping": metadata, "target_static_device_metadata": metadata,
                "source_group_path": str(cached["path"].resolve()), "source_group_sha256": sha256_file(cached["path"]),
                "sequence_constraints": {"min_events": cached["minimum"], "max_events": cached["maximum"]},
                "prompt": prompt, "prompt_sha256": prompt_hash, "schema_version": SCHEMA_VERSION,
                "data_boundary": boundary_declaration(), "uses_target_behavior": False,
            }
            requests.append(request)
            archive.joinpath(f"{request_id}.txt").write_text(prompt, encoding="utf-8")
    validate_codex56_requests(requests)
    if len(requests) != protocol["independent_request_count"]:
        raise ValueError("independent request count changed")
    if sum(item["candidate_sequence_count"] for item in requests) != protocol["candidate_pool_size"]:
        raise ValueError("candidate pool is not exactly 160")
    requests_path = output / "generation_requests.jsonl"
    requests_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8")
    manifest = {
        "experiment_identity": protocol["experiment_identity"], "generation_backend": BACKEND,
        "content_author": AUTHOR, "request_count": len(requests),
        "candidate_sequence_count": protocol["candidate_pool_size"], "final_sequence_count": protocol["final_sequence_count"],
        "programmatic_event_construction": False, "external_api_used": False, "api_key_used": False,
        "target_behavior_used": False, "gcad_source_enabled": False, "ranking_enabled": False,
    }
    (output / "request_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hashes = {
        "config": sha256_file(protocol_path), "preregistration_yaml": sha256_file(prereg / "preregistration.yaml"),
        "preregistration_json": sha256_file(prereg / "preregistration.json"),
        "source_support_plan_v2": sha256_file(support_path), "group_candidate_quotas": sha256_file(quota_path),
        "selector_config": sha256_file(selector_path), "generation_requests": sha256_file(requests_path),
    }
    (output / "pre_generation_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return requests


def _flatten(requests: list[dict], responses: list[dict]) -> list[dict]:
    request_map = {item["request_id"]: item for item in requests}
    metadata = requests[0]["target_static_device_metadata"]
    records = []
    for response in responses:
        request = request_map[response["request_id"]]
        for sequence in response["sequences"]:
            records.append({
                "sequence_id": sequence["sequence_id"], "request_id": response["request_id"],
                "group_id": request["group_id"], "events": sequence["events"],
                "actions": event_semantic_actions(sequence["events"], metadata),
                "devices": sorted({event["device"] for event in sequence["events"]}),
                "event_count": len(sequence["events"]), "content_author": AUTHOR,
                "programmatic_event_construction": False, "target_behavior_used": False,
            })
    return records


def candidate_sequence_support(records: list[dict]) -> Counter:
    """Count each action at most once per independently authored sequence."""
    return Counter(action for record in records for action in set(record["actions"]))


def validate_candidate_pool(directory: str | Path) -> dict:
    directory = Path(directory)
    provenance = json.loads((directory / "generation_provenance_gate.json").read_text(encoding="utf-8"))
    if provenance.get("passed") is not True:
        raise PermissionError("provenance must pass before candidate validation")
    requests = load_jsonl(directory / "generation_requests.jsonl")
    source_fingerprints = set()
    seen = set()
    from .source_copy_safe import action_sequence_fingerprint, numeric_action_sequence
    for request in requests:
        if request["source_group_path"] in seen:
            continue
        seen.add(request["source_group_path"])
        for sequence in pickle.loads(Path(request["source_group_path"]).read_bytes()):
            actions = numeric_action_sequence(sequence, request["dataset"])
            source_fingerprints.add(action_sequence_fingerprint([
                {"device": action.split(":", 1)[0], "action": action.split(":", 1)[1]} for action in actions
            ]))
    report = validate_responses(
        directory / "generation_requests.jsonl", directory / "generation_responses_raw.jsonl", directory,
        source_action_fingerprints=source_fingerprints, action_only_duplicate_check=True, raise_on_failure=False,
    )
    expected = sum(item["candidate_sequence_count"] for item in requests)
    legality_checks = {
        "json_zero": report["json_failures"] == 0, "schema_zero": report["schema_failures"] == 0,
        "provenance_metadata_zero": report["provenance_failures"] == 0,
        "illegal_device_zero": report["illegal_device_count"] == 0,
        "illegal_action_zero": report["illegal_action_count"] == 0,
        "count_complete": report["final_valid_sequence_count"] == expected == 160,
    }
    copy_checks = {"candidate_exact_duplicate_zero": report["duplicate_count"] == 0,
                   "source_representative_copy_zero": report["source_copy_count"] == 0}
    legality_gate = {"stage": "candidate_json_schema_static_legality", "passed": all(legality_checks.values()),
                     "checks": legality_checks, "uses_target_behavior": False}
    copy_gate = {"stage": "candidate_exact_duplicate_and_source_copy", "passed": all(copy_checks.values()),
                 "checks": copy_checks, "uses_target_behavior": False}
    (directory / "generation_legality_gate.json").write_text(json.dumps(legality_gate, indent=2) + "\n")
    (directory / "generation_copy_gate.json").write_text(json.dumps(copy_gate, indent=2) + "\n")
    if not legality_gate["passed"] or not copy_gate["passed"]:
        raise ValueError("formal candidate hard validation failed")
    responses = load_jsonl(directory / "generation_responses_validated.jsonl")
    (directory / "candidate_pool_response_records.jsonl").write_bytes(
        (directory / "generation_responses_validated.jsonl").read_bytes()
    )
    pool = _flatten(requests, responses)
    (directory / "candidate_pool.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in pool), encoding="utf-8"
    )
    return {"validation": report, "legality_gate": legality_gate, "copy_gate": copy_gate}


def audit_candidate_support(directory: str | Path, support_plan_path: str | Path, protocol_path: str | Path) -> dict:
    directory = Path(directory)
    plan = json.loads(Path(support_plan_path).read_text(encoding="utf-8"))
    validate_support_plan_v2(plan, set(plan["actions"]))
    policy = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    records = load_jsonl(directory / "candidate_pool.jsonl")
    support = candidate_sequence_support(records)
    groups = Counter(record["group_id"] for record in records)
    unsupported = sorted({
        (record["sequence_id"], action, record["group_id"])
        for record in records for action in set(record["actions"])
        if action in plan["actions"] and record["group_id"] not in plan["actions"][action]["allowed_source_groups"]
    })
    low = sorted(action for action in plan["actions"] if support[action] < plan["candidate_pool_sequence_support_target"])
    expected_groups = {group: item["candidate_quota"] for group, item in plan["groups"].items()}
    checks = {
        "candidate_count_160": len(records) == policy["candidate_pool_size"],
        "group_candidate_quotas": dict(groups) == expected_groups,
        "all_source_actions_meet_candidate_support_target": not low,
        "source_actions_only_in_supported_groups": not unsupported,
        "all_candidates_codex_authored": all(record.get("content_author") == AUTHOR for record in records),
        "programmatic_event_construction_absent": all(record.get("programmatic_event_construction") is False for record in records),
    }
    result = {
        "stage": "candidate_support_plan_audit", "passed": all(checks.values()), "checks": checks,
        "candidate_count": len(records), "candidate_action_sequence_support": dict(sorted(support.items())),
        "low_support_actions": low,
        "unsupported_source_action_group_usages": [
            {"sequence_id": sid, "action": action, "group_id": group} for sid, action, group in unsupported
        ],
        "group_candidate_counts": dict(sorted(groups.items())), "uses_target_behavior": False,
    }
    (directory / "candidate_support_report.json").write_text(json.dumps(result, indent=2) + "\n")
    (directory / "candidate_support_gate.json").write_text(json.dumps({
        "stage": result["stage"], "passed": result["passed"], "checks": checks, "uses_target_behavior": False,
    }, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError("candidate support-plan audit failed")
    return result


def _source_records(requests: list[dict], metadata: dict) -> list[list[str]]:
    records = []
    seen = set()
    for request in requests:
        path = request["source_group_path"]
        if path in seen:
            continue
        seen.add(path)
        records.extend(numeric_semantic_actions(sequence, request["dataset"], metadata)
                       for sequence in pickle.loads(Path(path).read_bytes()))
    return records


def _selection_constraints(records: list[dict], *, plan: dict, protocol: dict,
                           source_sequences: list[list[str]], source_transitions: set[tuple[str, str]]) -> dict:
    supports = candidate_sequence_support(records)
    groups = Counter(record["group_id"] for record in records)
    tokens = Counter(action for record in records for action in record["actions"])
    source_actions = set(plan["actions"])
    target_only_tokens = sum(count for action, count in tokens.items() if action not in source_actions)
    transitions = [pair for record in records for pair in zip(record["actions"], record["actions"][1:])]
    transition_coverage = sum(pair in source_transitions for pair in transitions) / max(1, len(transitions))
    source_metrics = sequence_metrics(source_sequences)
    selected_metrics = sequence_metrics([record["actions"] for record in records])
    distribution = {
        "unique_sequence_ratio": selected_metrics["unique_sequence_ratio"] >= min(0.98, source_metrics["unique_sequence_ratio"]),
        "lengths_vary": selected_metrics["length"]["distinct_count"] >= min(3, source_metrics["length"]["distinct_count"]),
        "bigram_entropy_source_relative": selected_metrics["normalized_bigram_entropy"] >= 0.8 * source_metrics["normalized_bigram_entropy"],
        "trigram_entropy_source_relative": selected_metrics["normalized_trigram_entropy"] >= 0.8 * source_metrics["normalized_trigram_entropy"],
        "top_template_share": selected_metrics["top_1_template_share"] <= max(0.1, source_metrics["top_1_template_share"]),
        "maximum_action_share": selected_metrics["maximum_action_share"] <= max(0.25, 1.5 * source_metrics["maximum_action_share"]),
    }
    final_quotas = {group: item["final_quota"] for group, item in plan["groups"].items()}
    source_devices = {action.split(":", 1)[0] for action in source_actions}
    selected_devices = {device for record in records for device in record["devices"]}
    checks = {
        "group_quotas_not_below_final": all(groups[group] >= count for group, count in final_quotas.items()),
        "selected_action_support_target": all(supports[action] >= plan["selected_sequence_support_target"] for action in source_actions),
        "source_vocabulary_recall_one": source_actions <= set(tokens),
        "metadata_only_share": target_only_tokens / max(1, sum(tokens.values())) <= protocol["source_semantic_v2"]["maximum_target_metadata_only_token_share"],
        "length_coverage": selected_metrics["length"]["distinct_count"] >= 3,
        "source_device_coverage": source_devices <= selected_devices,
        "transition_coverage": transition_coverage >= protocol["source_semantic_v2"]["minimum_source_transition_coverage"],
        **distribution,
    }
    return {"passed": all(checks.values()), "checks": checks, "action_support": dict(sorted(supports.items())),
            "group_counts": dict(sorted(groups.items())), "transition_coverage": transition_coverage,
            "metadata_only_token_share": target_only_tokens / max(1, sum(tokens.values()))}


def select_support_preserving_subset(directory: str | Path, support_plan_path: str | Path,
                                     protocol_path: str | Path, source_full_path: str | Path) -> dict:
    directory = Path(directory)
    support_gate = json.loads((directory / "candidate_support_gate.json").read_text(encoding="utf-8"))
    if support_gate.get("passed") is not True:
        raise PermissionError("candidate support audit must pass before selection")
    protocol = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(support_plan_path).read_text(encoding="utf-8"))
    frozen_selector = json.loads((directory / "selector_config.json").read_text(encoding="utf-8"))
    if frozen_selector != protocol["selector"]:
        raise ValueError("selector configuration changed after generation")
    requests = load_jsonl(directory / "generation_requests.jsonl")
    responses = load_jsonl(directory / "candidate_pool_response_records.jsonl")
    records = _flatten(requests, responses)
    metadata = requests[0]["target_static_device_metadata"]
    source_numeric = pickle.loads(Path(source_full_path).read_bytes())
    source_sequences = [numeric_semantic_actions(sequence, requests[0]["dataset"], metadata) for sequence in source_numeric]
    source_transitions = {pair for sequence in source_sequences for pair in zip(sequence, sequence[1:])}
    current = list(records)
    rejected = []
    trace = []
    final_quotas = {group: item["final_quota"] for group, item in plan["groups"].items()}
    while len(current) > protocol["final_sequence_count"]:
        supports = candidate_sequence_support(current)
        groups = Counter(record["group_id"] for record in current)
        ranked = []
        for index, record in enumerate(current):
            surplus = groups[record["group_id"]] - final_quotas[record["group_id"]]
            if surplus <= 0:
                continue
            trial = current[:index] + current[index + 1:]
            constraints = _selection_constraints(
                trial, plan=plan, protocol=protocol, source_sequences=source_sequences,
                source_transitions=source_transitions,
            )
            if not constraints["passed"]:
                continue
            redundancy = min((supports[action] for action in set(record["actions"])), default=0)
            similarity = max((_sequence_similarity(tuple(record["actions"]), tuple(other["actions"]))
                              for other in current if other["sequence_id"] != record["sequence_id"]), default=0.0)
            transition_contribution = sum(pair in source_transitions for pair in zip(record["actions"], record["actions"][1:]))
            priority = (-redundancy, -surplus, -similarity, transition_contribution, record["sequence_id"])
            ranked.append((priority, index, constraints, redundancy, surplus, similarity, transition_contribution))
        if not ranked:
            failure = {
                "stage": "deterministic_support_preserving_selection", "passed": False,
                "selected_count_at_failure": len(current), "supplemental_generation_allowed": False,
                "event_modification_allowed": False, "uses_target_behavior": False,
            }
            (directory / "selection_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise ValueError("no feasible deterministic deletion; replicate-5 selection failed")
        priority, index, constraints, redundancy, surplus, similarity, transition_contribution = min(ranked)
        removed = current.pop(index)
        rejected.append(removed)
        trace.append({
            "deletion_step": len(trace) + 1, "sequence_id": removed["sequence_id"],
            "group_id": removed["group_id"], "reason": "highest frozen deterministic removal priority among feasible candidates",
            "priority": {"minimum_action_support_before_deletion": redundancy, "group_surplus_before_deletion": surplus,
                         "maximum_candidate_similarity": similarity, "observed_transition_contribution": transition_contribution,
                         "sequence_id_tie_break": removed["sequence_id"]},
            "post_deletion_constraints": constraints, "target_result_used": False,
            "reconstruction_loss_used": False, "event_content_modified": False,
        })
    final = _selection_constraints(current, plan=plan, protocol=protocol,
                                   source_sequences=source_sequences, source_transitions=source_transitions)
    final_groups = Counter(record["group_id"] for record in current)
    exact_group_quotas = all(final_groups[group] == count for group, count in final_quotas.items())
    expected_rejections = protocol["candidate_pool_size"] - protocol["final_sequence_count"]
    passed = (len(current) == protocol["final_sequence_count"] and len(rejected) == expected_rejections
              and final["passed"] and exact_group_quotas)
    if not passed:
        raise ValueError("deterministic selector did not produce a formally valid 137-sequence set")
    selected_ids = {record["sequence_id"] for record in current}
    selected_responses = []
    for response in responses:
        kept = [sequence for sequence in response["sequences"] if sequence["sequence_id"] in selected_ids]
        if kept:
            selected_responses.append({**response, "sequences": kept})
    (directory / "selection_trace.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in trace), encoding="utf-8"
    )
    (directory / "selected_137.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in current), encoding="utf-8"
    )
    (directory / "rejected_23.jsonl").write_text(
        "".join(json.dumps({**item, "selection_reason": trace[index]["reason"]}, ensure_ascii=False) + "\n"
                for index, item in enumerate(rejected)), encoding="utf-8"
    )
    selected_path = directory / "generation_responses_selected.jsonl"
    selected_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected_responses), encoding="utf-8")
    report = {
        "stage": "deterministic_support_preserving_selection", "passed": True,
        "candidate_count": len(records), "selected_count": len(current), "rejected_count": len(rejected),
        "final_group_quotas_exact": exact_group_quotas, "final_constraints": final,
        "event_content_modified": False, "supplemental_generation_used": False,
        "target_behavior_used": False, "reconstruction_loss_used": False,
        "selection_trace_sha256": sha256_file(directory / "selection_trace.jsonl"),
        "selected_sha256": sha256_file(selected_path),
    }
    (directory / "final_support_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "selection_gate.json").write_text(json.dumps({
        "stage": report["stage"], "passed": True, "event_content_modified": False,
        "uses_target_behavior": False,
    }, indent=2) + "\n")
    return report


def require_v5_gate(directory: str | Path, name: str) -> None:
    gate = json.loads((Path(directory) / name).read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("uses_target_behavior") is not False:
        raise PermissionError(f"required v5 gate failed: {name}")


def require_pre_tof_v5(directory: str | Path) -> None:
    for name in (
        "generation_provenance_gate.json", "generation_legality_gate.json", "generation_copy_gate.json",
        "candidate_support_gate.json", "selection_gate.json", "generation_quality_gate.json",
        "source_semantic_v2_pre_tof_gate.json", "split_feasibility_pre_tof_gate.json",
    ):
        require_v5_gate(directory, name)
