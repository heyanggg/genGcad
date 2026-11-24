"""Reproducible relation artifact preparation; reads source training only."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

try:
    from .action_semantics import build_action_semantics, canonical_sha256
    from .experiment import atomic_write_json
    from .relation_adapter import adapt_relations
except ImportError:
    from action_semantics import build_action_semantics, canonical_sha256
    from experiment import atomic_write_json
    from relation_adapter import adapt_relations


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_relation_stage(source_path, dataset, source_environment,
                           target_environment, actions, devices, gss, gcad,
                           output_path, *, mode="adapted", min_support=3,
                           max_relationships=12, max_per_target=4):
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.name != "split_trn.pkl":
        raise ValueError("relation stage requires the TSS source split_trn.pkl")
    if source_path.parent.name != source_environment or source_path.parent.parent.name != dataset:
        raise ValueError("source training path does not match dataset and source environment")
    module_root = Path(__file__).resolve().parent
    implementation = {name: file_sha256(module_root / name) for name in (
        "action_semantics.py", "relation_adapter.py", "relation_guidance.py",
        "relation_stage.py", "text_translation_matrix.py",
    )}
    semantics = build_action_semantics(actions, devices)
    source_hash = file_sha256(source_path)
    config = {"mode": mode, "min_support": min_support,
              "max_relationships": max_relationships, "max_per_target": max_per_target}
    inputs = {
        "source_training": {"path": str(source_path.resolve()), "sha256": source_hash,
                            "dataset": dataset, "environment": source_environment},
        "gss_sha256": canonical_sha256(gss), "gcad_sha256": canonical_sha256(gcad),
        "semantics_sha256": canonical_sha256(semantics),
        "dictionary_sha256": semantics["dictionary_sha256"],
        "implementation_sha256": implementation,
    }
    fingerprint = canonical_sha256({"inputs": inputs, "config": config,
                                    "target_environment": target_environment})
    with source_path.open("rb") as handle:
        raw_sequences = pickle.load(handle)
    artifact = adapt_relations(
        gss, gcad, semantics, raw_sequences, dataset=dataset,
        source_environment=source_environment, target_environment=target_environment,
        source_sha256=source_hash, **config,
    )
    artifact.update(
        artifact_fingerprint=fingerprint, inputs=inputs, config=config,
        action_semantics=semantics, gss_snapshot=gss, gcad_snapshot=gcad,
        limitations=[
            "Action semantics are a frozen lexical-rule baseline; no LLM embeddings are implemented.",
            "Seed stability is not evidence of cross-environment invariance or causality.",
            "Source time bins cannot recover exact elapsed time or elapsed weeks.",
            "No target training, validation, test, attack data or labels are read by this stage.",
            "No action substitution, source augmentation, or learned regenerator is enabled.",
        ],
    )
    # Run-scoped artifacts must not silently change during resume. A code or input
    # change needs a new run ID rather than reusing responses under new evidence.
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != artifact:
            raise ValueError("relation artifact changed; use a new run ID or preview output directory")
    else:
        atomic_write_json(output_path, artifact)
    return artifact
