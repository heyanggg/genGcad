from __future__ import annotations

import hashlib
import json
import math
import pickle
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from SmartGen import dictionary
from SmartGen.gcad_source.cell_builder import numeric_to_text
from SmartGen.gcad_source.data_boundary import boundary_declaration
from SmartGen.gcad_source.prompt_adapter import build_original_smartgen_prompt
from SmartGen.gcad_source.semantic_channels import is_valid_semantic_channel

from .schemas import SCHEMA_VERSION, prompt_sha256


def derive_source_length_summary(sequences: list[list[int]]) -> dict:
    lengths = np.asarray([len(sequence) // 4 for sequence in sequences], dtype=float)
    if not len(lengths):
        raise ValueError("source representative group is empty")
    q10, q90 = np.percentile(lengths, [10, 90])
    allowed_min = max(2, int(math.floor(q10)))
    # The official prompt forbids singleton output. Preserve source upper-tail
    # information while guaranteeing room for at least three possible lengths.
    allowed_max = min(10, max(allowed_min + 2, int(math.ceil(q90))))
    return {
        "min": int(lengths.min()),
        "median": float(statistics.median(lengths)),
        "max": int(lengths.max()),
        "q10": float(q10),
        "q90": float(q90),
        "allowed_min": allowed_min,
        "allowed_max": allowed_max,
        "derivation": "max(2,floor(source_q10)); min(10,max(allowed_min+2,ceil(source_q90)))",
    }


def derive_group_semantic_envelope(sequences: list[list[int]], dataset: str) -> dict:
    inverse = {value: key for key, value in getattr(dictionary, f"{dataset}_actions").items()}
    action_sequences = [
        [
            inverse[int(sequence[index])]
            for index in range(3, len(sequence), 4)
            if is_valid_semantic_channel(inverse[int(sequence[index])])
        ]
        for sequence in sequences
    ]
    action_sequences = [sequence for sequence in action_sequences if sequence]
    actions = sorted({action for sequence in action_sequences for action in sequence})
    transitions = sorted({pair for sequence in action_sequences for pair in zip(sequence, sequence[1:])})
    if not actions:
        raise ValueError("source representative group has no valid semantic actions")
    return {
        "source_group_action_vocabulary": actions,
        "source_group_transition_vocabulary": [list(pair) for pair in transitions],
        "required_group_anchor_actions_per_sequence": 1,
        "minimum_batch_group_action_coverage": 0.5,
        "minimum_batch_group_transition_coverage": 0.2 if transitions else 0.0,
        "maximum_novel_action_types": len(actions),
        "derivation": "valid actions and adjacent transitions from this source SPPC group only",
        "uses_target_behavior": False,
    }


def _balanced_chunks(total: int, maximum: int = 20) -> list[int]:
    count = max(1, math.ceil(total / maximum))
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def _representative_id(group_id: str, index: int, sequence: list[int]) -> str:
    material = json.dumps(sequence, separators=(",", ":"))
    return f"rep_{group_id}_{index:03d}_{hashlib.sha256(material.encode()).hexdigest()[:10]}"


def _context_sentence(source_context: str, target_context: str) -> str:
    if target_context == "spring":
        return f"The previous environment is {source_context}. The changed environment is warm {target_context}."
    if target_context == "night":
        return (
            f"The previous environment: user is active during the {source_context} and rest at {target_context}. "
            f"The changed environment: user is active at {target_context} and rest during the {source_context}."
        )
    if target_context == "multiple":
        return (
            f"The previous environment was for a {source_context} person to be at home, and the changed "
            f"environment is for {target_context} people to be at home"
        )
    raise ValueError(f"unsupported target context: {target_context}")


def _assert_v2_output(output: Path) -> None:
    if "codex_generation_v2" not in output.parts:
        raise ValueError("grouped A2 requests must be isolated under outputs/codex_generation_v2")


def export_grouped_baseline_requests(
    output_dir: str | Path,
    *,
    experiment_id: str,
    dataset: str,
    source_context: str,
    target_context: str,
    compression_threshold: float,
    group_plan_path: str | Path,
    target_metadata_path: str | Path,
    original_gss_path: str | Path,
    device_control_path: str | Path,
    source_root: str | Path = "SmartGen/IoT_data",
    replicate: int = 1,
    source_copy_safe_config_path: str | Path | None = None,
) -> list[dict]:
    output = Path(output_dir)
    _assert_v2_output(output)
    archive = output / "prompt_archive"
    archive.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(group_plan_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(target_metadata_path).read_text(encoding="utf-8"))
    gss = json.loads(Path(original_gss_path).read_text(encoding="utf-8"))
    device_control = Path(device_control_path).read_text(encoding="utf-8")
    copy_safe = None
    copy_safe_config_sha = None
    denylist_sha = None
    if source_copy_safe_config_path is not None:
        from .source_copy_safe import (
            build_source_denylist,
            load_frozen_config,
            sha256_file,
            write_source_denylist,
        )

        copy_safe_path = Path(source_copy_safe_config_path).resolve()
        copy_safe = load_frozen_config(copy_safe_path)
        copy_safe_config_sha = sha256_file(copy_safe_path)
        denylist = build_source_denylist(
            plan,
            dataset=dataset,
            source_context=source_context,
            compression_threshold=compression_threshold,
            source_root=source_root,
        )
        _, denylist_sha = write_source_denylist(output, denylist)
    requests = []
    group_manifest = []
    global_batch = 1
    for group in plan["groups"]:
        group_id = str(group["group_id"])
        source_path = (
            Path(source_root) / dataset / source_context
            / f"trn_day_{group_id}_SPPC_th={compression_threshold}.pkl"
        )
        with source_path.open("rb") as handle:
            numeric = pickle.load(handle)
        text_sequences = numeric_to_text(numeric, dataset)
        representative_ids = [
            _representative_id(group_id, index, sequence)
            for index, sequence in enumerate(numeric)
        ]
        length_summary = derive_source_length_summary(numeric)
        semantic_envelope = derive_group_semantic_envelope(numeric, dataset)
        base_prompt = build_original_smartgen_prompt(
            device_control,
            _context_sentence(source_context, target_context),
            text_sequences,
            gss,
        )
        diversity = (
            f"\nGrouped generation protocol: This request uses only source group {group_id}. "
            f"Generate exactly {{count}} mutually distinct sequences, each with {length_summary['allowed_min']} to "
            f"{length_summary['allowed_max']} events. Use varied lengths across the returned set; do not fix every "
            "sequence at four or five events. Avoid repeated full templates, shared fixed openings or endings, and "
            "mechanical coverage of every transition. Maintain diverse legal behavior combinations without inventing "
            "a device-action pair outside the supplied static mapping. Source-semantic constraints: every generated "
            "sequence must contain at least one action from the source-group action vocabulary below; across this "
            "response, at least half of all events must use that vocabulary; use observed source-group adjacent "
            "transitions where applicable, while never copying a complete representative sequence; and introduce no "
            "more novel action types than the number of source-group action types.\n"
            f"Source-group action vocabulary: {json.dumps(semantic_envelope['source_group_action_vocabulary'])}\n"
            f"Observed source-group transitions: {json.dumps(semantic_envelope['source_group_transition_vocabulary'])}\n"
            "Return only the required structured data."
        )
        if copy_safe is not None:
            diversity += (
                "\nSource-copy-safe-v1 hard constraint: Do not reproduce, event by event, the complete ordered "
                "device:action sequence of any source representative. Day or hour changes do not make a copied "
                "device:action sequence new. An exact source-representative action-sequence copy is a hard generation "
                "invalidity. Automatic replacement is allowed only for exact source copies, invalid JSON/schema, "
                "illegal devices or actions, missing requested sequences, or exact duplicates within the generated "
                "set. Never replace or select an otherwise legal candidate because of a source-semantic score or any "
                "target result."
            )
        chunks = _balanced_chunks(int(group["requested_sequence_count"]), 20)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        group_manifest.append({
            "group_id": group_id,
            "source_date_or_partition": group.get("source_date_or_partition", group_id.split("_", 1)[0]),
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha,
            "representative_sequence_count": len(numeric),
            "representative_sequence_ids": representative_ids,
            "source_length_summary": length_summary,
            "source_semantic_envelope": semantic_envelope,
            "requested_sequence_count": int(group["requested_sequence_count"]),
            "request_chunks": chunks,
        })
        for group_batch, count in enumerate(chunks, 1):
            prompt = base_prompt + diversity.format(count=count)
            prompt_hash = prompt_sha256(prompt)
            material = f"{experiment_id}|{group_id}|{group_batch}|{replicate}|{prompt_hash}"
            request_id = "req_" + hashlib.sha256(material.encode()).hexdigest()[:20]
            request = {
                "request_id": request_id,
                "experiment_id": experiment_id,
                "dataset": dataset,
                "context": target_context,
                "method": "baseline_v2",
                "canonical_generation_method": "baseline",
                "replicate": replicate,
                "generation_batch": global_batch,
                "group_batch": group_batch,
                "group_id": group_id,
                "source_date_or_partition": group.get("source_date_or_partition", group_id.split("_", 1)[0]),
                "representative_sequence_ids": representative_ids,
                "representative_sequence_count": len(numeric),
                "source_length_summary": length_summary,
                "source_semantic_envelope": semantic_envelope,
                "source_group_path": str(source_path.resolve()),
                "source_group_sha256": source_sha,
                "requested_sequence_count": count,
                "group_requested_sequence_count": int(group["requested_sequence_count"]),
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "generation_backend": "codex_agent_file",
                "source_sequence_count": len(numeric),
                "target_context_description": {"change": f"{source_context} to {target_context}", "static_only": True},
                "target_static_device_metadata": metadata,
                "original_gss_path": str(Path(original_gss_path).resolve()),
                "fused_gss_path": None,
                "stable_relation_path": None,
                "sequence_constraints": {
                    "min_events": length_summary["allowed_min"],
                    "max_events": length_summary["allowed_max"],
                    "lengths_must_vary_within_group": len(chunks) == 1 and count > 1,
                    "minimum_group_anchor_actions_per_sequence": 1,
                    "minimum_batch_group_action_coverage": 0.5,
                    "minimum_batch_group_transition_coverage": semantic_envelope[
                        "minimum_batch_group_transition_coverage"
                    ],
                    "maximum_novel_action_types": semantic_envelope["maximum_novel_action_types"],
                },
                "schema_version": SCHEMA_VERSION,
                "data_boundary": boundary_declaration(),
                "uses_target_behavior": False,
            }
            if copy_safe is not None:
                request["source_copy_safe"] = {
                    "version": copy_safe["version"],
                    "config_path": str(copy_safe_path),
                    "config_sha256": copy_safe_config_sha,
                    "source_denylist_sha256": denylist_sha,
                    "maximum_replacement_candidates": copy_safe["maximum_replacement_candidates"],
                    "uses_target_behavior": False,
                }
            requests.append(request)
            (archive / f"{request_id}.txt").write_text(prompt, encoding="utf-8")
            global_batch += 1
    output.joinpath("generation_requests.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8"
    )
    manifest = {
        "experiment_id": experiment_id,
        "protocol": "SmartGen-compatible grouped Codex-file baseline v2",
        "official_backend_reproduced": False,
        "group_count": len(group_manifest),
        "request_count": len(requests),
        "requested_sequence_count": sum(item["requested_sequence_count"] for item in requests),
        "groups": group_manifest,
        "generation_backend": "codex_agent_file",
        "external_api": False,
        "api_key_required": False,
        "target_behavior_read": False,
        "source_semantic_policy": "source_semantic_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if copy_safe is not None:
        manifest["source_copy_safe"] = {
            "version": copy_safe["version"],
            "config_path": str(copy_safe_path),
            "config_sha256": copy_safe_config_sha,
            "source_denylist_sha256": denylist_sha,
            "maximum_replacement_candidates": copy_safe["maximum_replacement_candidates"],
            "uses_target_behavior": False,
        }
    output.joinpath("request_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if copy_safe is not None:
        from .source_copy_safe import sha256_file

        protocol = {
            "version": copy_safe["version"],
            "status": "frozen_before_generation",
            "config_path": str(copy_safe_path),
            "config_sha256": copy_safe_config_sha,
            "source_denylist_sha256": denylist_sha,
            "prompt_prohibits_exact_source_copy": True,
            "exact_source_copy_is_hard_invalidity": True,
            "allowed_automatic_replacement_categories": list(
                copy_safe["allowed_automatic_replacement_categories"]
            ),
            "maximum_replacement_candidates": copy_safe["maximum_replacement_candidates"],
            "reuse_replicate_2_valid_samples": False,
            "replacement_by_source_semantic_score": False,
            "replacement_by_target_result": False,
            "uses_target_behavior": False,
        }
        protocol_path = output / "source_copy_safe_protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        checksums = {
            "protocol": "source-copy-safe-v1",
            "frozen_before_generation": True,
            "sha256": {
                "generation_requests.jsonl": sha256_file(output / "generation_requests.jsonl"),
                "source_copy_safe_protocol.json": sha256_file(protocol_path),
                "source_copy_safe_config": copy_safe_config_sha,
                "source_representative_denylist.json": denylist_sha,
            },
        }
        output.joinpath("pre_generation_checksums.json").write_text(
            json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
        )
    return requests
