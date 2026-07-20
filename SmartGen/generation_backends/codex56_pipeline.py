from __future__ import annotations

import json
import pickle
from pathlib import Path

from .codex56_provenance import sha256_file
from .response_loader import load_jsonl
from .source_copy_safe import action_sequence_fingerprint, numeric_action_sequence
from .validation import validate_responses


def validate_codex56_candidates(directory: str | Path) -> dict:
    directory = Path(directory)
    provenance = json.loads((directory / "generation_provenance_gate.json").read_text(encoding="utf-8"))
    if provenance.get("passed") is not True:
        raise ValueError("provenance must pass before candidate validation")
    requests = load_jsonl(directory / "generation_requests.jsonl")
    source_fingerprints = set()
    seen = set()
    dataset = requests[0].get("dataset", "fr")
    for request in requests:
        path = request["source_group_path"]
        if path in seen:
            continue
        seen.add(path)
        for sequence in pickle.loads(Path(path).read_bytes()):
            actions = numeric_action_sequence(sequence, dataset)
            events = [{"device": item.split(":", 1)[0], "action": item.split(":", 1)[1]} for item in actions]
            source_fingerprints.add(action_sequence_fingerprint(events))
    report = validate_responses(
        directory / "generation_requests.jsonl",
        directory / "generation_responses_raw.jsonl",
        directory,
        source_action_fingerprints=source_fingerprints,
        action_only_duplicate_check=True,
        raise_on_failure=False,
    )
    legality_checks = {
        "json_zero": report["json_failures"] == 0,
        "schema_zero": report["schema_failures"] == 0,
        "provenance_metadata_zero": report["provenance_failures"] == 0,
        "illegal_device_zero": report["illegal_device_count"] == 0,
        "illegal_action_zero": report["illegal_action_count"] == 0,
        "count_complete": report["final_valid_sequence_count"] == 137,
    }
    copy_checks = {
        "generated_exact_duplicate_zero": report["duplicate_count"] == 0,
        "source_representative_copy_zero": report["source_copy_count"] == 0,
    }
    legality_gate = {
        "stage": "json_schema_static_legality",
        "passed": all(legality_checks.values()),
        "checks": legality_checks,
        "uses_target_behavior": False,
    }
    copy_gate = {
        "stage": "exact_duplicate_and_source_copy",
        "passed": all(copy_checks.values()),
        "checks": copy_checks,
        "uses_target_behavior": False,
    }
    (directory / "generation_legality_gate.json").write_text(json.dumps(legality_gate, indent=2) + "\n")
    (directory / "generation_copy_gate.json").write_text(json.dumps(copy_gate, indent=2) + "\n")
    if not legality_gate["passed"] or not copy_gate["passed"]:
        raise ValueError("formal Codex candidate validation failed")
    validated = directory / "generation_responses_validated.jsonl"
    selected = directory / "generation_responses_selected.jsonl"
    selected.write_bytes(validated.read_bytes())
    selection = {
        "candidate_count": 137,
        "selected_count": 137,
        "selection_rule": "select all hard-valid candidates; oversampling ratio 1.0",
        "candidate_sha256": sha256_file(validated),
        "selected_sha256": sha256_file(selected),
        "event_content_modified": False,
        "favorable_subset_selection": False,
        "uses_target_behavior": False,
    }
    (directory / "candidate_selection_report.json").write_text(json.dumps(selection, indent=2) + "\n")
    return {"validation": report, "legality_gate": legality_gate, "copy_gate": copy_gate, "selection": selection}


def require_gate(directory: str | Path, name: str) -> None:
    gate = json.loads((Path(directory) / name).read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("uses_target_behavior") is not False:
        raise PermissionError(f"formal v4 gate failed or lacks zero-target declaration: {name}")


def require_pre_tof_v4(directory: str | Path) -> None:
    for name in (
        "generation_provenance_gate.json",
        "generation_legality_gate.json",
        "generation_copy_gate.json",
        "generation_quality_gate.json",
        "source_semantic_v2_pre_tof_gate.json",
        "split_feasibility_pre_tof_gate.json",
    ):
        require_gate(directory, name)


def require_pre_training_v4(directory: str | Path) -> None:
    require_pre_tof_v4(directory)
    for name in ("source_semantic_v2_post_tof_gate.json", "split_feasibility_post_tof_gate.json"):
        require_gate(directory, name)
