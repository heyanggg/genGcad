from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .response_loader import load_jsonl
from .semantic_actions import (
    classify_action_spaces,
    event_semantic_actions,
    numeric_semantic_actions,
    static_legal_actions,
)


def load_protocol(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _request_metadata(request: dict) -> dict[str, list[str]]:
    return request.get("target_static_device_action_mapping", request.get("target_static_device_metadata"))


def _records_from_responses(requests: list[dict], responses: list[dict], metadata: dict) -> list[dict]:
    request_map = {item["request_id"]: item for item in requests}
    records = []
    for response in responses:
        request = request_map[response["request_id"]]
        for sequence in response["sequences"]:
            records.append({
                "sequence_id": sequence["sequence_id"],
                "group_id": request["group_id"],
                "actions": event_semantic_actions(sequence["events"], metadata),
                "devices": sorted({event["device"] for event in sequence["events"]}),
                "event_count": len(sequence["events"]),
            })
    return records


def _quantiles(values: list[int]) -> dict:
    array = np.asarray(values or [0], dtype=float)
    return {
        "minimum": int(array.min()), "q10": float(np.percentile(array, 10)),
        "q25": float(np.percentile(array, 25)), "median": float(np.median(array)),
    }


def evaluate_source_semantic_v2(
    records: list[dict],
    *,
    source_numeric: list[list[int]],
    group_source_numeric: dict[str, list[list[int]]],
    dataset: str,
    metadata: dict[str, list[str]],
    policy: dict,
) -> dict:
    spaces = classify_action_spaces(source_numeric, dataset, metadata)
    source_actions = set(spaces["source_observed_target_legal_actions"])
    target_only = set(spaces["target_metadata_only_actions"])
    legal = static_legal_actions(metadata)
    source_token_support = Counter(spaces["source_token_support"])
    source_sequence_support = Counter(spaces["source_sequence_support"])
    generated_tokens = Counter(action for record in records for action in record["actions"])
    generated_sequence_support = Counter(
        action for record in records for action in set(record["actions"])
    )
    generated_group_support: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for action in set(record["actions"]):
            generated_group_support[action].add(record["group_id"])
    generated_actions = set(generated_tokens)
    source_covered = source_actions & generated_actions
    total_generated_tokens = sum(generated_tokens.values())
    precision = sum(generated_tokens[action] for action in source_actions) / max(1, total_generated_tokens)
    global_recall = len(source_covered) / max(1, len(source_actions))
    weighted_recall = sum(source_token_support[action] for action in source_covered) / max(
        1, sum(source_token_support.values())
    )
    rare_actions = {
        action for action, support in source_sequence_support.items()
        if support <= policy["rare_source_sequence_support_maximum"]
    }
    rare_recall = len(rare_actions & generated_actions) / max(1, len(rare_actions))

    group_reports = {}
    group_recalls = []
    group_transition_recalls = []
    source_transitions = Counter()
    source_transition_sequence_support = Counter()
    for sequence in source_numeric:
        actions = numeric_semantic_actions(sequence, dataset, metadata)
        transitions = list(zip(actions, actions[1:]))
        source_transitions.update(transitions)
        source_transition_sequence_support.update(set(transitions))
    generated_by_group = defaultdict(list)
    for record in records:
        generated_by_group[record["group_id"]].append(record["actions"])
    for group, numeric in sorted(group_source_numeric.items()):
        source_group_sequences = [numeric_semantic_actions(item, dataset, metadata) for item in numeric]
        source_group_vocab = {action for sequence in source_group_sequences for action in sequence}
        generated_group_vocab = {
            action for sequence in generated_by_group.get(group, []) for action in sequence
        }
        group_recall = len(source_group_vocab & generated_group_vocab) / max(1, len(source_group_vocab))
        source_group_transitions = {
            pair for sequence in source_group_sequences for pair in zip(sequence, sequence[1:])
        }
        generated_group_transitions = {
            pair for sequence in generated_by_group.get(group, []) for pair in zip(sequence, sequence[1:])
        }
        transition_recall = len(source_group_transitions & generated_group_transitions) / max(
            1, len(source_group_transitions)
        )
        group_recalls.append(group_recall)
        group_transition_recalls.append(transition_recall)
        group_reports[group] = {
            "source_vocabulary_count": len(source_group_vocab),
            "generated_source_vocabulary_count": len(source_group_vocab & generated_group_vocab),
            "source_vocabulary_recall": group_recall,
            "source_transition_count": len(source_group_transitions),
            "source_transition_recall": transition_recall,
        }

    generated_transition_tokens = Counter(
        pair for record in records for pair in zip(record["actions"], record["actions"][1:])
    )
    generated_transition_sequence_support = Counter(
        pair for record in records for pair in set(zip(record["actions"], record["actions"][1:]))
    )
    total_generated_transitions = sum(generated_transition_tokens.values())
    observed_transition_tokens = sum(
        count for pair, count in generated_transition_tokens.items() if pair in source_transitions
    )
    low_source_transition_tokens = sum(
        count for pair, count in generated_transition_tokens.items()
        if source_transition_sequence_support[pair] < policy["minimum_source_transition_sequence_support"]
    )
    per_sequence_transition_minimum = []
    for record in records:
        pairs = list(zip(record["actions"], record["actions"][1:]))
        per_sequence_transition_minimum.append(
            min((source_transition_sequence_support[pair] for pair in pairs), default=0)
        )

    support_values = [generated_sequence_support[action] for action in sorted(generated_actions)]
    low_support = sorted(
        action for action in generated_actions
        if generated_sequence_support[action] < policy["minimum_sequence_support_per_action"]
    )
    per_action = {
        action: {
            "category": "source_observed_target_legal" if action in source_actions else "target_metadata_only",
            "token_occurrence_count": generated_tokens[action],
            "sequence_support_count": generated_sequence_support[action],
            "group_support_count": len(generated_group_support[action]),
            "source_token_support_count": source_token_support[action],
            "source_sequence_support_count": source_sequence_support[action],
        }
        for action in sorted(generated_actions)
    }
    target_only_used = generated_actions & target_only
    metrics = {
        "sequence_count": len(records),
        "source_observed_target_legal_action_count": len(source_actions),
        "target_metadata_only_action_count": len(target_only),
        "generated_action_vocabulary_count": len(generated_actions),
        "source_action_precision": precision,
        "global_source_vocabulary_recall": global_recall,
        "group_aware_source_vocabulary_recall": float(np.mean(group_recalls)),
        "frequency_weighted_source_vocabulary_recall": weighted_recall,
        "rare_source_action_count": len(rare_actions),
        "rare_action_recall": rare_recall,
        "static_metadata_action_coverage": len(generated_actions & legal) / max(1, len(legal)),
        "target_metadata_only_used_vocabulary_count": len(target_only_used),
        "target_metadata_only_token_share": sum(generated_tokens[action] for action in target_only) / max(
            1, total_generated_tokens
        ),
        "target_metadata_only_sequence_support": sum(
            any(action in target_only for action in record["actions"]) for record in records
        ),
        "target_metadata_only_group_support": len({
            record["group_id"] for record in records if any(action in target_only for action in record["actions"])
        }),
        "per_action_sequence_support_summary": {
            **_quantiles(support_values),
            "support_le_1_count": sum(value <= 1 for value in support_values),
            "support_le_3_count": sum(value <= 3 for value in support_values),
            "support_le_5_count": sum(value <= 5 for value in support_values),
            "low_support_action_count": len(low_support),
            "low_support_action_ratio": len(low_support) / max(1, len(generated_actions)),
        },
        "low_support_actions": low_support,
        "source_transition_coverage": observed_transition_tokens / max(1, total_generated_transitions),
        "unseen_transition_ratio": 1.0 - observed_transition_tokens / max(1, total_generated_transitions),
        "low_support_transition_ratio": low_source_transition_tokens / max(1, total_generated_transitions),
        "group_transition_recall": float(np.mean(group_transition_recalls)),
        "minimum_per_sequence_transition_source_support": min(per_sequence_transition_minimum, default=0),
    }
    thresholds = {
        key: policy[key] for key in (
            "minimum_source_action_precision", "minimum_global_source_vocabulary_recall",
            "minimum_group_vocabulary_recall", "minimum_frequency_weighted_recall",
            "minimum_rare_action_recall", "minimum_sequence_support_per_action",
            "maximum_low_support_action_ratio", "maximum_target_metadata_only_token_share",
            "minimum_source_transition_coverage", "maximum_unseen_transition_ratio",
        )
    }
    checks = {
        "source_action_precision": precision >= policy["minimum_source_action_precision"],
        "global_source_vocabulary_recall": global_recall >= policy["minimum_global_source_vocabulary_recall"],
        "group_vocabulary_recall": metrics["group_aware_source_vocabulary_recall"]
        >= policy["minimum_group_vocabulary_recall"],
        "frequency_weighted_recall": weighted_recall >= policy["minimum_frequency_weighted_recall"],
        "rare_action_recall": rare_recall >= policy["minimum_rare_action_recall"],
        "per_action_sequence_support": not low_support,
        "low_support_action_ratio": metrics["per_action_sequence_support_summary"]["low_support_action_ratio"]
        <= policy["maximum_low_support_action_ratio"],
        "target_metadata_only_token_share": metrics["target_metadata_only_token_share"]
        <= policy["maximum_target_metadata_only_token_share"],
        "source_transition_coverage": metrics["source_transition_coverage"]
        >= policy["minimum_source_transition_coverage"],
        "unseen_transition_ratio": metrics["unseen_transition_ratio"] <= policy["maximum_unseen_transition_ratio"],
    }
    return {
        "version": "source_semantic_v2",
        "policy": policy,
        "action_spaces": spaces,
        "thresholds": thresholds,
        "metrics": metrics,
        "per_action_support": per_action,
        "transition_support": {
            "source_transition_token_support": {
                " -> ".join(pair): count for pair, count in sorted(source_transitions.items())
            },
            "source_transition_sequence_support": {
                " -> ".join(pair): count for pair, count in sorted(source_transition_sequence_support.items())
            },
            "generated_transition_sequence_support": {
                " -> ".join(pair): count for pair, count in sorted(generated_transition_sequence_support.items())
            },
            "per_sequence_minimum_source_support": per_sequence_transition_minimum,
        },
        "group_reports": group_reports,
        "gate": {
            "stage": "source_semantic_v2",
            "passed": all(checks.values()),
            "checks": checks,
            "uses_target_behavior": False,
        },
        "uses_target_behavior": False,
    }


def run_source_semantic_v2(
    directory: str | Path,
    dataset: str,
    source_path: str | Path,
    protocol_path: str | Path,
    *,
    stage: str = "pre_tof",
    records: list[dict] | None = None,
) -> dict:
    directory = Path(directory)
    requests = load_jsonl(directory / "generation_requests.jsonl")
    metadata = _request_metadata(requests[0])
    if records is None:
        response_path = directory / "generation_responses_selected.jsonl"
        if not response_path.exists():
            response_path = directory / "generation_responses_validated.jsonl"
        responses = load_jsonl(response_path)
        records = _records_from_responses(requests, responses, metadata)
    source_numeric = pickle.loads(Path(source_path).read_bytes())
    group_source_numeric = {}
    for request in requests:
        if request["group_id"] not in group_source_numeric:
            group_source_numeric[request["group_id"]] = pickle.loads(Path(request["source_group_path"]).read_bytes())
    policy = load_protocol(protocol_path)["source_semantic_v2"]
    result = evaluate_source_semantic_v2(
        records,
        source_numeric=source_numeric,
        group_source_numeric=group_source_numeric,
        dataset=dataset,
        metadata=metadata,
        policy=policy,
    )
    result["stage"] = stage
    result["gate"]["stage"] = f"{stage}_source_semantic_v2"
    (directory / f"source_semantic_v2_{stage}_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (directory / f"source_semantic_v2_{stage}_gate.json").write_text(
        json.dumps(result["gate"], indent=2) + "\n", encoding="utf-8"
    )
    return result
