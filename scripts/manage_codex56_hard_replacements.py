#!/usr/bin/env python3
"""Serialize direct agent edits and record diffs; never invent event content."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequence_map(records: list[dict]) -> dict[str, dict]:
    return {sequence["sequence_id"]: sequence for record in records for sequence in record["sequences"]}


def prepare(directory: Path) -> None:
    raw = directory / "generation_responses_raw.jsonl"
    initial = directory / "generation_responses_initial_raw.jsonl"
    failures = directory / "generation_failures.jsonl"
    initial_failures = directory / "generation_failures_initial.jsonl"
    if not initial.exists():
        shutil.copyfile(raw, initial)
    if not initial_failures.exists():
        shutil.copyfile(failures, initial_failures)
    (directory / "generation_responses_agent_working.json").write_text(
        json.dumps(load_jsonl(raw), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def finalize(directory: Path) -> None:
    initial_records = load_jsonl(directory / "generation_responses_initial_raw.jsonl")
    working_records = json.loads((directory / "generation_responses_agent_working.json").read_text(encoding="utf-8"))
    before = sequence_map(initial_records)
    after = sequence_map(working_records)
    failures = load_jsonl(directory / "generation_failures_initial.jsonl")
    reason_by_id = {item["sequence_id"]: item for item in failures}
    changed = sorted(sequence_id for sequence_id in before if before[sequence_id] != after.get(sequence_id))
    if set(changed) != set(reason_by_id):
        raise ValueError("direct edits must match exactly the initial hard-invalid sequence IDs")
    mapping = []
    for sequence_id in changed:
        failure = reason_by_id[sequence_id]
        if failure["category"] not in {"duplicate", "source_copy"}:
            raise ValueError("replacement category is not permitted")
        mapping.append({
            "sequence_id": sequence_id,
            "request_id": failure["request_id"],
            "reason": failure["category"],
            "failure_detail": failure["reason"],
            "old_events": before[sequence_id]["events"],
            "new_events": after[sequence_id]["events"],
            "content_author": "codex_gpt56_agent",
            "source_semantic_score_used": False,
            "target_result_used": False,
        })
    if len(mapping) > 10:
        raise ValueError("frozen maximum replacement count exceeded")
    (directory / "generation_responses_raw.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in working_records),
        encoding="utf-8",
    )
    (directory / "replacement_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "finalize"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else finalize)(args.directory)


if __name__ == "__main__":
    main()
