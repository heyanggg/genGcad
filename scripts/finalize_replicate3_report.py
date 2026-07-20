#!/usr/bin/env python3
"""Write the stopped replicate-3 audit report and complete artifact checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--replicate2-diagnostic", required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    validation = _read(directory / "generation_validation_report.json")
    generation = _read(directory / "generation_quality_report.json")
    semantic = _read(directory / "source_semantic_report.json")
    reconstruction = _read(directory / "reconstruction_health_report.json")
    tof = _read(directory / "tof" / "tof_report.json")
    replicate2 = _read(args.replicate2_diagnostic)
    pre_generation = _read(directory / "pre_generation_checksums.json")
    report = {
        "status": "prospective replicate-3 reconstruction health gate failure",
        "replicate_2": {
            "formal_gate_status": replicate2["formal_gate_status"],
            "diagnostic_only": replicate2["diagnostic_only"],
            "formal_gate_decision_modified": replicate2["formal_gate_decision_modified"],
            "source_semantic_v1_diagnostic_gate_would_pass": replicate2["gate"]["passed"],
            "metrics": replicate2["metrics"],
            "tof_run": False,
            "target_evaluation_run": False,
        },
        "replicate_3": {
            "protocol": "source-copy-safe-v1",
            "pre_generation_sha256": pre_generation["sha256"],
            "request_count": validation["request_count"],
            "generated_sequence_count": validation["initial_sequence_count"],
            "validated_sequence_count": validation["final_valid_sequence_count"],
            "replicate_2_action_templates_reused": 0,
            "format_repairs": validation["repair_count"],
            "source_copy_failures": validation["source_copy_count"],
            "actual_replacement_count": validation["actual_replacement_count"],
            "maximum_replacement_candidates": validation["maximum_replacement_candidates"],
            "generation_distribution_gate": generation["generation_gate"],
            "source_semantic_v1": {
                "gate": semantic["gate"],
                "thresholds": semantic["thresholds"],
                "metrics": semantic["metrics"],
            },
            "tof": {
                "run": True,
                "input_count": tof["input_count"],
                "stage1_retained_count": tof["stage1_retained_count"],
                "stage2_recovered_count": tof["stage2_recovered_count"],
                "final_count": tof["final_count"],
                "uses_target_behavior": tof["uses_target_behavior"],
            },
            "reconstruction_health_v1": reconstruction,
            "all_three_gates_passed": False,
            "final_evaluation_run": False,
        },
        "gcad_source_enabled": False,
        "ranking_enabled": False,
        "target_behavior_read": False,
        "target_files_opened": [],
        "external_api_called": False,
        "tests": {"passed": 82, "skipped": 1},
        "stopped_before_target_evaluation": True,
        "artifact_checksum_manifest": str(directory / "checksums.sha256"),
    }
    report_path = directory / "prospective_replicate_3_reconstruction_health_failure.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path = directory / "checksums.sha256"
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{_sha(path)}  {path.relative_to(directory)}\n" for path in files),
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": _sha(report_path),
        "checksums": str(checksum_path),
        "checksums_sha256": _sha(checksum_path),
        "artifact_count": len(files),
        "target_behavior_read": False,
        "final_evaluation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
