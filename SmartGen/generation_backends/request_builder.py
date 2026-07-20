from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from SmartGen.gcad_source.data_boundary import boundary_declaration, require_roles
from SmartGen.gcad_source.data_roles import RoleBoundPath

from .schemas import ALLOWED_METHODS, SCHEMA_VERSION, canonical_generation_method, prompt_sha256


def _request_id(experiment_id: str, method: str, replicate: int, batch: int, prompt_hash: str) -> str:
    material = f"{experiment_id}|{canonical_generation_method(method)}|{replicate}|{batch}|{prompt_hash}"
    return "req_" + hashlib.sha256(material.encode()).hexdigest()[:20]


def export_requests(
    output_dir: str | Path,
    *,
    experiment_id: str,
    dataset: str,
    context: str,
    method: str,
    replicate: int,
    prompt: str,
    requested_sequence_count: int,
    batch_size: int,
    source_sequence_count: int,
    target_context_description: dict,
    target_static_device_metadata: dict,
    original_gss_path: str,
    fused_gss_path: str | None = None,
    stable_relation_path: str | None = None,
    min_events: int = 2,
    max_events: int = 10,
    artifacts: list[RoleBoundPath] | None = None,
) -> list[dict]:
    if artifacts:
        require_roles("generation", artifacts)
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported method: {method}")
    if requested_sequence_count < 1 or not 10 <= batch_size <= 25:
        raise ValueError("requested count must be positive and generation batch_size must be 10..25")
    if method in {"baseline", "ranking"} and (fused_gss_path or stable_relation_path):
        raise ValueError("baseline/ranking requests cannot reference GCAD artifacts")
    prompt_hash = prompt_sha256(prompt)
    output = Path(output_dir)
    archive = output / "prompt_archive"
    archive.mkdir(parents=True, exist_ok=True)
    requests = []
    remaining = requested_sequence_count
    batch = 1
    while remaining:
        count = min(batch_size, remaining)
        request_id = _request_id(experiment_id, method, replicate, batch, prompt_hash)
        request = {
            "request_id": request_id,
            "experiment_id": experiment_id,
            "dataset": dataset,
            "context": context,
            "method": method,
            "canonical_generation_method": canonical_generation_method(method),
            "replicate": replicate,
            "generation_batch": batch,
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "requested_sequence_count": count,
            "source_sequence_count": source_sequence_count,
            "target_context_description": target_context_description,
            "target_static_device_metadata": target_static_device_metadata,
            "original_gss_path": original_gss_path,
            "fused_gss_path": fused_gss_path,
            "stable_relation_path": stable_relation_path,
            "sequence_constraints": {"min_events": min_events, "max_events": max_events},
            "schema_version": SCHEMA_VERSION,
            "data_boundary": boundary_declaration(),
        }
        requests.append(request)
        (archive / f"{request_id}.txt").write_text(prompt, encoding="utf-8")
        remaining -= count
        batch += 1
    request_path = output / "generation_requests.jsonl"
    request_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "context": context,
        "method": method,
        "canonical_generation_method": canonical_generation_method(method),
        "replicate": replicate,
        "request_count": len(requests),
        "requested_sequence_count": requested_sequence_count,
        "batch_size": batch_size,
        "prompt_sha256": prompt_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_backend": "codex_agent_file",
        "external_api": False,
        "api_key_required": False,
        "data_boundary": boundary_declaration(),
    }
    (output / "request_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return requests

