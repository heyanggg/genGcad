from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from .response_loader import load_jsonl
from .semantic_actions import event_semantic_actions
from .source_semantic_v2 import load_protocol, _request_metadata


def _length_bin(length: int) -> str:
    if length <= 3:
        return "short"
    if length <= 5:
        return "medium"
    return "long"


def response_records(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    requests = load_jsonl(directory / "generation_requests.jsonl")
    response_path = directory / "generation_responses_selected.jsonl"
    if not response_path.exists():
        response_path = directory / "generation_responses_validated.jsonl"
    responses = load_jsonl(response_path)
    request_map = {item["request_id"]: item for item in requests}
    metadata = _request_metadata(requests[0])
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


def post_tof_records(directory: str | Path, tof_path: str | Path) -> list[dict]:
    directory = Path(directory)
    records = response_records(directory)
    generated = pickle.loads((directory / "generated_sequences.pkl").read_bytes())
    tof = pickle.loads(Path(tof_path).read_bytes())
    if len(generated) != len(records):
        raise ValueError("generated PKL cannot be mapped to response provenance")
    buckets: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for sequence, record in zip(generated, records):
        buckets[tuple(sequence)].append(record)
    selected = []
    for sequence in tof:
        bucket = buckets.get(tuple(sequence), [])
        if not bucket:
            raise ValueError("TOF sequence lacks an authored response provenance record")
        selected.append(bucket.pop(0))
    return selected


def _labels(record: dict) -> set[str]:
    return {
        f"group={record['group_id']}",
        f"length={_length_bin(record['event_count'])}",
        *(f"action={action}" for action in set(record["actions"])),
        *(f"device={device}" for device in set(record["devices"])),
    }


def _tie(seed: int, sequence_id: str) -> int:
    return int(hashlib.sha256(f"{seed}|{sequence_id}".encode()).hexdigest(), 16)


def build_coverage_constrained_split(records: list[dict], policy: dict, seed: int) -> dict:
    ids = [record["sequence_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("sequence IDs must be unique before splitting")
    ratio = float(policy["train_validation_ratio"])
    validation_target = len(records) - int(len(records) * ratio)
    minimum_train = int(policy["minimum_train_sequence_support_for_validation_action"])
    total_action_support = Counter(action for record in records for action in set(record["actions"]))
    total_labels = Counter(label for record in records for label in _labels(record))
    validation_label_counts = Counter()
    remaining_train_support = total_action_support.copy()
    selected: set[int] = set()
    validation_fraction = 1.0 - ratio

    def feasible(index: int) -> bool:
        return all(
            remaining_train_support[action] - 1 >= minimum_train
            for action in set(records[index]["actions"])
        )

    def utility(index: int) -> tuple[float, int]:
        value = 0.0
        for label in _labels(records[index]):
            desired = max(1.0, total_labels[label] * validation_fraction)
            deficit = desired - validation_label_counts[label]
            weight = 3.0 if label.startswith("action=") else 1.0
            value += weight * deficit / desired
        return value, -_tie(seed, records[index]["sequence_id"])

    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["group_id"]].append(index)
    for group, indices in sorted(groups.items()):
        if len(indices) < policy["minimum_group_size_for_validation"]:
            continue
        candidates = [index for index in indices if feasible(index)]
        if not candidates:
            continue
        choice = max(candidates, key=utility)
        selected.add(choice)
        for action in set(records[choice]["actions"]):
            remaining_train_support[action] -= 1
        validation_label_counts.update(_labels(records[choice]))

    while len(selected) < validation_target:
        candidates = [index for index in range(len(records)) if index not in selected and feasible(index)]
        if not candidates:
            break
        choice = max(candidates, key=utility)
        selected.add(choice)
        for action in set(records[choice]["actions"]):
            remaining_train_support[action] -= 1
        validation_label_counts.update(_labels(records[choice]))

    validation_indices = sorted(selected)
    train_indices = [index for index in range(len(records)) if index not in selected]
    train_actions = Counter(action for index in train_indices for action in set(records[index]["actions"]))
    validation_actions = Counter(action for index in validation_indices for action in set(records[index]["actions"]))
    train_groups = Counter(records[index]["group_id"] for index in train_indices)
    validation_groups = Counter(records[index]["group_id"] for index in validation_indices)
    unseen = sorted(action for action in validation_actions if train_actions[action] == 0)
    low_support = sorted(action for action in validation_actions if train_actions[action] < minimum_train)
    all_lengths = Counter(_length_bin(record["event_count"]) for record in records)
    val_lengths = Counter(_length_bin(records[index]["event_count"]) for index in validation_indices)
    length_tv = 0.5 * sum(
        abs(val_lengths[key] / max(1, len(validation_indices)) - all_lengths[key] / max(1, len(records)))
        for key in set(all_lengths) | set(val_lengths)
    )
    missing_train_groups = sorted(set(groups) - set(train_groups))
    missing_validation_groups = sorted(
        group for group, indices in groups.items()
        if len(indices) >= policy["minimum_group_size_for_validation"] and validation_groups[group] == 0
    )
    checks = {
        "target_validation_count_reached": len(validation_indices) == validation_target,
        "sequence_id_overlap_zero": not ({records[index]["sequence_id"] for index in train_indices}
        & {records[index]["sequence_id"] for index in validation_indices}),
        "all_groups_covered_in_train": not missing_train_groups,
        "eligible_groups_covered_in_validation": not missing_validation_groups,
        "validation_unseen_action_zero": not unseen,
        "validation_low_support_action_zero": not low_support,
        "length_distribution": length_tv <= policy["maximum_length_distribution_total_variation"],
    }
    return {
        "version": "split_feasibility_v1",
        "seed": seed,
        "train_validation_ratio": ratio,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_sequence_ids": [records[index]["sequence_id"] for index in train_indices],
        "validation_sequence_ids": [records[index]["sequence_id"] for index in validation_indices],
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "sequence_id_overlap_count": 0,
        "train_group_support": dict(sorted(train_groups.items())),
        "validation_group_support": dict(sorted(validation_groups.items())),
        "missing_train_groups": missing_train_groups,
        "missing_validation_groups": missing_validation_groups,
        "train_action_sequence_support": dict(sorted(train_actions.items())),
        "validation_action_sequence_support": dict(sorted(validation_actions.items())),
        "validation_unseen_actions": unseen,
        "validation_unseen_action_count": len(unseen),
        "validation_low_support_actions": low_support,
        "validation_low_support_action_count": len(low_support),
        "minimum_train_support_for_validation_actions": min(
            (train_actions[action] for action in validation_actions), default=0
        ),
        "length_distribution_total_variation": length_tv,
        "checks": checks,
        "passed": all(checks.values()),
        "uses_target_behavior": False,
    }


def run_split_feasibility(
    directory: str | Path,
    protocol_path: str | Path,
    *,
    stage: str = "pre_tof",
    records: list[dict] | None = None,
) -> dict:
    directory = Path(directory)
    records = records or response_records(directory)
    policy = load_protocol(protocol_path)["split_feasibility_v1"]
    manifests = [
        build_coverage_constrained_split(records, policy, int(seed)) for seed in policy["fixed_split_seeds"]
    ]
    result = {
        "version": "split_feasibility_v1",
        "stage": stage,
        "policy": policy,
        "split_manifests": manifests,
        "gate": {
            "stage": f"{stage}_split_feasibility_v1",
            "passed": all(item["passed"] for item in manifests),
            "all_fixed_seeds_required": True,
            "favorable_seed_selection_allowed": False,
            "uses_target_behavior": False,
        },
        "uses_target_behavior": False,
    }
    for manifest in manifests:
        (directory / f"split_manifest_{stage}_seed{manifest['seed']}.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    (directory / f"split_feasibility_{stage}_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (directory / f"split_feasibility_{stage}_gate.json").write_text(
        json.dumps(result["gate"], indent=2) + "\n", encoding="utf-8"
    )
    return result


def materialize_manifest_split(
    sequence_file: str | Path,
    manifest_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
) -> dict:
    sequences = pickle.loads(Path(sequence_file).read_bytes())
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if max(manifest["train_indices"] + manifest["validation_indices"], default=-1) >= len(sequences):
        raise ValueError("split manifest index exceeds frozen sequence set")
    Path(train_path).write_bytes(pickle.dumps([sequences[index] for index in manifest["train_indices"]]))
    Path(validation_path).write_bytes(pickle.dumps([sequences[index] for index in manifest["validation_indices"]]))
    return manifest
