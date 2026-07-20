#!/usr/bin/env python3
"""Build the historical programmatic replicate-3 stress-test plan.

This authoring helper reads only frozen generation requests, the source-copy
denylist, and replicate-2 responses used exclusively to enforce the no-reuse
protocol. It does not read target-normal or attack behavior and does not score
or rank candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from SmartGen.generation_backends.response_loader import load_jsonl
from SmartGen.generation_backends.source_copy_safe import verify_source_copy_safe_artifacts


def _template_from_response(sequence: dict) -> tuple[str, ...]:
    return tuple(
        event["action"] if ":" in event["action"] else f"{event['device']}:{event['action']}"
        for event in sequence["events"]
    )


def _load_replicate2_templates(path: Path) -> set[tuple[str, ...]]:
    return {
        _template_from_response(sequence)
        for response in load_jsonl(path)
        for sequence in response["sequences"]
    }


def _candidate(vocabulary: list[str], transitions: list[list[str]], length: int, serial: int, variant: int):
    size = len(vocabulary)
    actions = [
        vocabulary[(serial * 5 + variant * 3 + position * 7 + position * position) % size]
        for position in range(length)
    ]
    if transitions:
        transition = transitions[(serial * 7 + variant * 5) % len(transitions)]
        position = (serial + variant) % (length - 1)
        actions[position:position + 2] = transition
    return tuple(actions)


def build_plan(directory: Path, replicate2_raw: Path) -> list[dict]:
    verify_source_copy_safe_artifacts(directory)
    requests = load_jsonl(directory / "generation_requests.jsonl")
    denylist = json.loads((directory / "source_representative_denylist.json").read_text(encoding="utf-8"))
    forbidden_source = {tuple(item["actions"]) for item in denylist["entries"]}
    forbidden_replicate2 = _load_replicate2_templates(replicate2_raw)
    used: set[tuple[str, ...]] = set()
    plan = []
    global_serial = 0
    for request_index, request in enumerate(requests):
        envelope = request["source_semantic_envelope"]
        vocabulary = list(envelope["source_group_action_vocabulary"])
        transitions = list(envelope["source_group_transition_vocabulary"])
        minimum = max(3, int(request["sequence_constraints"]["min_events"]))
        maximum = int(request["sequence_constraints"]["max_events"])
        if maximum < minimum:
            raise ValueError("copy-safe authoring requires room for at least three events")
        sequences = []
        for sequence_index in range(request["requested_sequence_count"]):
            length = minimum + (global_serial + request_index) % (maximum - minimum + 1)
            for variant in range(1000):
                actions = _candidate(vocabulary, transitions, length, global_serial + request_index, variant)
                if actions not in used and actions not in forbidden_source and actions not in forbidden_replicate2:
                    break
            else:
                raise RuntimeError("could not author a new legal source-copy-safe action sequence")
            used.add(actions)
            sequences.append({
                "sequence_id": f"r3_{request['group_id']}_{request['group_batch']}_{sequence_index + 1:02d}",
                "day": (global_serial * 3 + request_index) % 7,
                "start_bin": (global_serial * 5 + request_index * 2) % 8,
                "actions": [action.replace(":", "|", 1) for action in actions],
            })
            global_serial += 1
        plan.append({
            "request_id": request["request_id"],
            "group_id": request["group_id"],
            "sequences": sequences,
        })
    if global_serial != 137:
        raise ValueError(f"frozen request total changed: expected 137, got {global_serial}")
    return plan


def main() -> None:
    print(
        "WARNING: This script creates programmatic source-constrained compositions.\n"
        "Its outputs are not Codex/LLM-authored generations and must not be used as the formal Codex baseline."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--replicate2-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_plan(args.directory, args.replicate2_raw)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "request_count": len(plan),
        "sequence_count": sum(len(item["sequences"]) for item in plan),
        "replicate_2_action_templates_reused": 0,
        "source_representative_action_templates_copied": 0,
        "uses_target_behavior": False,
    }, indent=2))


if __name__ == "__main__":
    main()
