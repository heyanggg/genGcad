from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path

import yaml

from SmartGen.gcad_source.cell_builder import numeric_to_text
from SmartGen.gcad_source.data_boundary import boundary_declaration
from SmartGen.gcad_source.prompt_adapter import build_original_smartgen_prompt

from .codex56_provenance import BACKEND, validate_codex56_requests
from .grouped_requests import _balanced_chunks, _context_sentence, derive_source_length_summary
from .schemas import SCHEMA_VERSION, prompt_sha256
from .semantic_actions import classify_action_spaces, numeric_semantic_actions


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _length_quota(count: int, minimum: int, maximum: int) -> dict[str, int]:
    lengths = list(range(minimum, maximum + 1))
    quotient, remainder = divmod(count, len(lengths))
    return {str(length): quotient + (index < remainder) for index, length in enumerate(lengths)}


def export_codex56_grouped_requests(
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
    replicate: int = 4,
) -> list[dict]:
    output = Path(output_dir)
    preregistration = Path(preregistration_dir)
    output.mkdir(parents=True, exist_ok=True)
    preregistration.mkdir(parents=True, exist_ok=True)
    archive = output / "prompt_archive"
    archive.mkdir(exist_ok=True)
    protocol_text = Path(protocol_path).read_text(encoding="utf-8")
    protocol = yaml.safe_load(protocol_text)
    if protocol["generation_backend"] != BACKEND or protocol["programmatic_event_construction"] is not False:
        raise ValueError("invalid formal Codex v4 protocol")
    plan = json.loads(Path(group_plan_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(target_metadata_path).read_text(encoding="utf-8"))
    gss = json.loads(Path(original_gss_path).read_text(encoding="utf-8"))
    device_control = Path(device_control_path).read_text(encoding="utf-8")
    source_full = pickle.loads(Path(source_full_path).read_bytes())
    action_spaces = classify_action_spaces(source_full, dataset, metadata)
    minimum_support = protocol["source_semantic_v2"]["minimum_sequence_support_per_action"]
    support_plan = {
        "version": "source_support_plan_v4",
        "role": "batch-level guidance for the Codex GPT-5.6 agent; never an event-construction plan",
        "source_observed_target_legal_actions": action_spaces["source_observed_target_legal_actions"],
        "target_metadata_only_actions": action_spaces["target_metadata_only_actions"],
        "global_minimum_sequence_support_per_source_action": minimum_support,
        "maximum_target_metadata_only_token_share": protocol["source_semantic_v2"][
            "maximum_target_metadata_only_token_share"
        ],
        "groups": [],
        "contains_complete_event_sequences": False,
        "contains_per_sequence_actions": False,
        "contains_combination_formula": False,
        "programmatic_event_construction": False,
        "uses_target_behavior": False,
    }
    requests = []
    global_batch = 1
    group_cache = {}
    for group in plan["groups"]:
        group_id = str(group["group_id"])
        source_path = (
            Path(source_root) / dataset / source_context
            / f"trn_day_{group_id}_SPPC_th={compression_threshold}.pkl"
        )
        numeric = pickle.loads(source_path.read_bytes())
        group_sequences = [numeric_semantic_actions(sequence, dataset, metadata) for sequence in numeric]
        group_actions = sorted({action for sequence in group_sequences for action in sequence})
        group_transitions = sorted({pair for sequence in group_sequences for pair in zip(sequence, sequence[1:])})
        length_summary = derive_source_length_summary(numeric)
        minimum = max(protocol["length_policy"]["minimum_events"], length_summary["allowed_min"])
        maximum = min(protocol["length_policy"]["maximum_events"], length_summary["allowed_max"])
        if minimum > maximum:
            maximum = minimum
        group_cache[group_id] = {
            "source_path": source_path,
            "numeric": numeric,
            "representatives": numeric_to_text(numeric, dataset),
            "actions": group_actions,
            "transitions": [list(pair) for pair in group_transitions],
            "length_summary": length_summary,
            "minimum": minimum,
            "maximum": maximum,
        }
        support_plan["groups"].append({
            "group_id": group_id,
            "requested_sequence_count": int(group["requested_sequence_count"]),
            "source_action_anchors": group_actions,
            "source_transition_hints": [list(pair) for pair in group_transitions],
            "minimum_distinct_source_actions_to_cover": math.ceil(
                protocol["source_semantic_v2"]["minimum_group_vocabulary_recall"] * len(group_actions)
            ),
            "length_distribution_quota": _length_quota(
                int(group["requested_sequence_count"]), minimum, maximum
            ),
        })

    support_plan_path = preregistration / "source_support_plan.json"
    support_plan_path.write_text(json.dumps(support_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preregistration.joinpath("preregistration.yaml").write_text(protocol_text, encoding="utf-8")
    preregistration.joinpath("preregistration.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for group in plan["groups"]:
        group_id = str(group["group_id"])
        cached = group_cache[group_id]
        base_prompt = build_original_smartgen_prompt(
            device_control,
            _context_sentence(source_context, target_context),
            cached["representatives"],
            gss,
        )
        chunks = _balanced_chunks(int(group["requested_sequence_count"]), 20)
        for group_batch, requested_count in enumerate(chunks, 1):
            candidate_count = int(requested_count * protocol["candidate_oversampling_ratio"])
            guidance = {
                "group_id": group_id,
                "source_action_anchors": cached["actions"],
                "source_transition_hints": cached["transitions"],
                "batch_candidate_count": candidate_count,
                "group_support_targets": next(
                    item for item in support_plan["groups"] if item["group_id"] == group_id
                ),
            }
            prompt = base_prompt + (
                "\n\nFormal A4 GPT-5.6 Codex-agent generation instructions:\n"
                f"Author exactly {candidate_count} independent candidate sequences for source SPPC group {group_id}. "
                f"Each sequence must contain {cached['minimum']} to {cached['maximum']} events. You, the active "
                "Codex agent, must create every event directly; no Python plan, template loop, rotation, Cartesian "
                "product, or GSS constructor may create event content. Reference sequences teach behavior structure "
                "but must never be copied action-for-action. Use only device/action pairs in the static legal mapping. "
                "Keep most actions connected to source-normal behavior, cover multiple source-observed actions across "
                "the batch, preserve plausible source transitions where natural, and add only a small number of "
                "reasonable new combinations. Do not let a rare action appear as an accidental one-off: important or "
                "rare source actions need support across multiple distinct sequences in the complete 137-candidate "
                "collection. Do not mechanically recite the support plan, do not use a fixed opening/ending template, "
                "and make every sequence independently plausible in warm spring. Day and hour changes do not make a "
                "copied source action sequence new. Return a JSON response record matching the frozen schema.\n"
                f"Batch-level support guidance (not an event plan): {json.dumps(guidance, ensure_ascii=False)}"
            )
            prompt_hash = prompt_sha256(prompt)
            request_id = "req4_" + hashlib.sha256(
                f"{experiment_id}|{group_id}|{group_batch}|{replicate}|{prompt_hash}".encode()
            ).hexdigest()[:20]
            request = {
                "request_id": request_id,
                "experiment_id": experiment_id,
                "dataset": dataset,
                "group_id": group_id,
                "group_batch": group_batch,
                "replicate": replicate,
                "generation_backend": BACKEND,
                "model_family": "GPT-5.6 via Codex agent",
                "requested_sequence_count": requested_count,
                "candidate_sequence_count": candidate_count,
                "representative_sequences": cached["representatives"],
                "source_action_anchors": cached["actions"],
                "source_transition_hints": cached["transitions"],
                "target_context_description": {
                    "change": f"{source_context} to {target_context}", "static_only": True
                },
                "target_static_device_action_mapping": metadata,
                "target_static_device_metadata": metadata,
                "group_support_targets": guidance["group_support_targets"],
                "source_group_path": str(cached["source_path"].resolve()),
                "source_group_sha256": _sha(cached["source_path"]),
                "sequence_constraints": {
                    "min_events": cached["minimum"], "max_events": cached["maximum"]
                },
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "schema_version": SCHEMA_VERSION,
                "data_boundary": boundary_declaration(),
                "uses_target_behavior": False,
            }
            requests.append(request)
            archive.joinpath(f"{request_id}.txt").write_text(prompt, encoding="utf-8")
            global_batch += 1
    validate_codex56_requests(requests)
    if len(requests) != protocol["independent_request_count"]:
        raise ValueError("frozen independent request count changed")
    if sum(item["candidate_sequence_count"] for item in requests) != protocol["final_sequence_count"]:
        raise ValueError("candidate pool must be frozen at exactly 137 for v4")
    requests_path = output / "generation_requests.jsonl"
    requests_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8"
    )
    manifest = {
        "experiment_identity": protocol["experiment_identity"],
        "generation_backend": BACKEND,
        "content_author": "codex_gpt56_agent",
        "request_count": len(requests),
        "candidate_sequence_count": sum(item["candidate_sequence_count"] for item in requests),
        "final_sequence_count": protocol["final_sequence_count"],
        "programmatic_event_construction": False,
        "external_api_used": False,
        "api_key_used": False,
        "target_behavior_used": False,
        "gcad_source_enabled": False,
        "ranking_enabled": False,
    }
    (output / "request_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hashes = {
        "config": _sha(protocol_path),
        "preregistration_yaml": _sha(preregistration / "preregistration.yaml"),
        "preregistration_json": _sha(preregistration / "preregistration.json"),
        "source_support_plan": _sha(support_plan_path),
        "generation_requests": _sha(requests_path),
    }
    (output / "pre_generation_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return requests
