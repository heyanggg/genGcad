from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from SmartGen import dictionary
from SmartGen.gcad_source.semantic_channels import is_valid_semantic_channel

from .response_loader import load_jsonl


POLICY = {
    "version": "source_semantic_v1",
    "action_floor_fraction_of_cross_day_source": 0.5,
    "transition_floor_fraction_of_cross_day_source": 0.25,
    "minimum_group_anchor_share": 0.5,
    "maximum_action_vocabulary_expansion": 2.0,
    "minimum_zero_anchor_cap": 0.1,
    "zero_anchor_source_multiplier": 5.0,
    "maximum_zero_anchor_cap": 0.25,
}


def _numeric_sequences(sequences: list[list[int]], dataset: str) -> tuple[list[list[str]], list[int]]:
    inverse = {value: key for key, value in getattr(dictionary, f"{dataset}_actions").items()}
    actions = []
    days = []
    for sequence in sequences:
        events = [
            (int(sequence[index]), inverse[int(sequence[index + 3])])
            for index in range(0, len(sequence), 4)
        ]
        events = [(day, action) for day, action in events if is_valid_semantic_channel(action)]
        if events:
            days.append(events[0][0])
            actions.append([action for _, action in events])
    return actions, days


def _coverage(sequence: list[str], vocabulary: set[str]) -> float:
    return sum(item in vocabulary for item in sequence) / len(sequence) if sequence else 0.0


def _transition_coverage(sequence: list[str], transitions: set[tuple[str, str]]) -> float:
    pairs = list(zip(sequence, sequence[1:]))
    return sum(pair in transitions for pair in pairs) / len(pairs) if pairs else 1.0


def cross_day_source_calibration(source_sequences: list[list[str]], source_days: list[int]) -> dict:
    by_day = defaultdict(list)
    for day, sequence in zip(source_days, source_sequences):
        by_day[day].append(sequence)
    action_coverages = []
    transition_coverages = []
    anchors = []
    per_day = {}
    for held_day, held_sequences in sorted(by_day.items()):
        training = [sequence for day, values in by_day.items() if day != held_day for sequence in values]
        vocabulary = {action for sequence in training for action in sequence}
        transitions = {pair for sequence in training for pair in zip(sequence, sequence[1:])}
        day_actions = [_coverage(sequence, vocabulary) for sequence in held_sequences]
        day_transitions = [
            _transition_coverage(sequence, transitions) for sequence in held_sequences if len(sequence) > 1
        ]
        day_anchors = [any(action in vocabulary for action in sequence) for sequence in held_sequences]
        action_coverages.extend(day_actions)
        transition_coverages.extend(day_transitions)
        anchors.extend(day_anchors)
        per_day[str(held_day)] = {
            "sequence_count": len(held_sequences),
            "mean_action_coverage": float(np.mean(day_actions)),
            "mean_transition_coverage": float(np.mean(day_transitions)) if day_transitions else 0.0,
            "zero_anchor_share": 1.0 - float(np.mean(day_anchors)),
        }
    return {
        "method": "deterministic leave-one-source-day-out coverage",
        "mean_action_coverage": float(np.mean(action_coverages)),
        "mean_transition_coverage": float(np.mean(transition_coverages)),
        "zero_anchor_share": 1.0 - float(np.mean(anchors)),
        "per_day": per_day,
    }


def semantic_support_report(
    generated_by_group: dict[str, list[list[str]]],
    source_sequences: list[list[str]],
    source_days: list[int],
    group_source_sequences: dict[str, list[list[str]]],
) -> dict:
    calibration = cross_day_source_calibration(source_sequences, source_days)
    source_vocabulary = {action for sequence in source_sequences for action in sequence}
    source_transitions = {pair for sequence in source_sequences for pair in zip(sequence, sequence[1:])}
    generated = [sequence for values in generated_by_group.values() for sequence in values]
    generated_vocabulary = {action for sequence in generated for action in sequence}
    action_coverages = [_coverage(sequence, source_vocabulary) for sequence in generated]
    transition_coverages = [_transition_coverage(sequence, source_transitions) for sequence in generated]
    global_anchors = [any(action in source_vocabulary for action in sequence) for sequence in generated]
    group_metrics = {}
    group_anchors = []
    per_sequence = []
    for group, sequences in sorted(generated_by_group.items()):
        group_vocabulary = {
            action for source_sequence in group_source_sequences[group] for action in source_sequence
        }
        coverages = [_coverage(sequence, group_vocabulary) for sequence in sequences]
        anchors = [any(action in group_vocabulary for action in sequence) for sequence in sequences]
        group_anchors.extend(anchors)
        group_metrics[group] = {
            "sequence_count": len(sequences),
            "source_group_action_vocabulary_count": len(group_vocabulary),
            "mean_group_action_coverage": float(np.mean(coverages)),
            "group_anchor_share": float(np.mean(anchors)),
            "zero_group_anchor_count": sum(not value for value in anchors),
        }
        for sequence, coverage, anchor in zip(sequences, coverages, anchors):
            per_sequence.append({
                "group_id": group,
                "length": len(sequence),
                "global_action_coverage": _coverage(sequence, source_vocabulary),
                "global_transition_coverage": _transition_coverage(sequence, source_transitions),
                "group_action_coverage": coverage,
                "has_global_source_anchor": any(action in source_vocabulary for action in sequence),
                "has_group_source_anchor": anchor,
            })
    thresholds = {
        "minimum_mean_global_action_coverage": (
            POLICY["action_floor_fraction_of_cross_day_source"] * calibration["mean_action_coverage"]
        ),
        "minimum_mean_global_transition_coverage": (
            POLICY["transition_floor_fraction_of_cross_day_source"] * calibration["mean_transition_coverage"]
        ),
        "maximum_zero_global_anchor_share": min(
            POLICY["maximum_zero_anchor_cap"],
            max(
                POLICY["minimum_zero_anchor_cap"],
                POLICY["zero_anchor_source_multiplier"] * calibration["zero_anchor_share"],
            ),
        ),
        "minimum_group_anchor_share": POLICY["minimum_group_anchor_share"],
        "maximum_action_vocabulary_expansion": POLICY["maximum_action_vocabulary_expansion"],
    }
    metrics = {
        "sequence_count": len(generated),
        "source_sequence_count": len(source_sequences),
        "source_action_vocabulary_count": len(source_vocabulary),
        "source_transition_count": len(source_transitions),
        "generated_action_vocabulary_count": len(generated_vocabulary),
        "generated_action_vocabulary_expansion": len(generated_vocabulary) / max(1, len(source_vocabulary)),
        "mean_global_action_coverage": float(np.mean(action_coverages)),
        "mean_global_transition_coverage": float(np.mean(transition_coverages)),
        "zero_global_anchor_count": sum(not value for value in global_anchors),
        "zero_global_anchor_share": 1.0 - float(np.mean(global_anchors)),
        "group_anchor_share": float(np.mean(group_anchors)),
    }
    checks = {
        "global_action_coverage": (
            metrics["mean_global_action_coverage"] >= thresholds["minimum_mean_global_action_coverage"]
        ),
        "global_transition_coverage": (
            metrics["mean_global_transition_coverage"] >= thresholds["minimum_mean_global_transition_coverage"]
        ),
        "zero_global_anchor_share": (
            metrics["zero_global_anchor_share"] <= thresholds["maximum_zero_global_anchor_share"]
        ),
        "group_anchor_share": metrics["group_anchor_share"] >= thresholds["minimum_group_anchor_share"],
        "action_vocabulary_expansion": (
            metrics["generated_action_vocabulary_expansion"]
            <= thresholds["maximum_action_vocabulary_expansion"]
        ),
    }
    return {
        "policy": POLICY,
        "calibration": calibration,
        "thresholds": thresholds,
        "metrics": metrics,
        "group_metrics": group_metrics,
        "per_sequence": per_sequence,
        "gate": {
            "stage": "pre_tof_source_semantic_coherence",
            "passed": all(checks.values()),
            "checks": checks,
            "threshold_basis": "source-only leave-one-day-out calibration plus fixed protocol tolerances",
            "uses_target_behavior": False,
        },
    }


def _assert_source_only_provenance(requests: list[dict], source_path: str | Path) -> None:
    source_parent = Path(source_path).resolve().parent
    if any(request.get("uses_target_behavior") is not False for request in requests):
        raise ValueError("generation request does not declare a zero-target boundary")
    group_parents = {Path(request["source_group_path"]).resolve().parent for request in requests}
    if group_parents != {source_parent}:
        raise ValueError("full source and group representatives must come from the same source-context directory")


def require_pre_tof_gates(directory: str | Path) -> None:
    directory = Path(directory)
    required = ("generation_quality_gate.json", "source_semantic_gate.json")
    for name in required:
        path = directory / name
        if not path.exists():
            raise ValueError(f"pre-TOF gate is missing: {name}")
        gate = json.loads(path.read_text(encoding="utf-8"))
        if gate.get("uses_target_behavior") is not False:
            raise ValueError(f"pre-TOF gate lacks zero-target declaration: {name}")
        if gate.get("passed") is not True:
            raise ValueError(f"pre-TOF gate failed: {name}")


def diagnose_source_semantics(directory: str | Path, dataset: str, source_path: str | Path) -> dict:
    directory = Path(directory)
    requests = load_jsonl(directory / "generation_requests.jsonl")
    responses = load_jsonl(directory / "generation_responses_validated.jsonl")
    _assert_source_only_provenance(requests, source_path)
    request_map = {request["request_id"]: request for request in requests}
    generated_by_group = defaultdict(list)
    for response in responses:
        group = request_map[response["request_id"]]["group_id"]
        for sequence in response["sequences"]:
            generated_by_group[group].append([
                event["action"] if ":" in event["action"] else f"{event['device']}:{event['action']}"
                for event in sequence["events"]
            ])
    source_numeric = pickle.loads(Path(source_path).read_bytes())
    source_sequences, source_days = _numeric_sequences(source_numeric, dataset)
    group_source_sequences = {}
    for request in requests:
        group = request["group_id"]
        if group not in group_source_sequences:
            numeric = pickle.loads(Path(request["source_group_path"]).read_bytes())
            group_source_sequences[group] = _numeric_sequences(numeric, dataset)[0]
    result = semantic_support_report(
        dict(generated_by_group), source_sequences, source_days, group_source_sequences
    )
    result.update({
        "protocol": "source-only semantic coherence audit",
        "dataset": dataset,
        "source_path": str(Path(source_path).resolve()),
        "raw_source_sequence_count": len(source_numeric),
        "usable_source_sequence_count": len(source_sequences),
        "invalid_semantic_events_removed": sum(len(sequence) // 4 for sequence in source_numeric)
        - sum(len(sequence) for sequence in source_sequences),
        "target_files_opened": [],
        "uses_target_behavior": False,
    })
    (directory / "source_semantic_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (directory / "source_semantic_gate.json").write_text(
        json.dumps(result["gate"] | {"thresholds": result["thresholds"], "metrics": result["metrics"]}, indent=2),
        encoding="utf-8",
    )
    return result
