"""Sparse semantic endpoint transport with separate GSS/GCAD scores.

Outputs are hypotheses derived from source evidence. Scores are neither target
observations nor probabilities. Environment labels alone provide no learned
preference score; the environment factor is only an explicit catalogue gate.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .action_semantics import canonical_sha256
    from .relation_adapter import _decode_events, _decode_gcad_events
except ImportError:
    from action_semantics import canonical_sha256
    from relation_adapter import _decode_events, _decode_gcad_events


TRANSFER_VERSION = "sparse-endpoint-transport-v1"
NON_TRANSFER_CAPABILITIES = {"notification", "refresh", "unknown"}


def load_embeddings(path, semantics, dataset):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in artifact.items() if k != "artifact_fingerprint"}
    if artifact.get("artifact_fingerprint") != canonical_sha256(payload):
        raise ValueError("embedding artifact fingerprint mismatch")
    if artifact.get("dataset") != dataset or artifact.get("dictionary_sha256") != semantics["dictionary_sha256"]:
        raise ValueError("embedding artifact dictionary or dataset mismatch")
    if artifact.get("semantics_sha256") != canonical_sha256(semantics):
        raise ValueError("embedding artifact semantic-label version mismatch")
    names, vectors = artifact.get("action_names", []), artifact.get("vectors", [])
    if len(names) != len(set(names)) or set(names) != set(semantics["records"]) or len(vectors) != len(names):
        raise ValueError("embedding action coverage mismatch")
    dimension = len(vectors[0]) if vectors else 0
    if not dimension or any(
        len(v) != dimension
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v)
        or abs(sum(x * x for x in v) - 1.0) > 1e-4 for v in vectors
    ):
        raise ValueError("embeddings must be finite, nonzero L2-normalized vectors of equal dimension")
    return artifact


def compatibility(left, right):
    """Conservative, independently visible mask; never infer a plug's appliance."""
    if not left["valid_device_action"] or not right["valid_device_action"]:
        return False, "non_device_action"
    if left["semantic_status"] != "rule_labeled" or right["semantic_status"] != "rule_labeled":
        return False, "unresolved_function_or_state"
    if left["capability"] in NON_TRANSFER_CAPABILITIES or right["capability"] in NON_TRANSFER_CAPABILITIES:
        return False, "telemetry_or_unknown_operation"
    if left["desired_state"] in {"unknown", "toggle"} or right["desired_state"] in {"unknown", "toggle"}:
        return False, "ambiguous_state_direction"
    if left["function"] != right["function"]:
        return False, "different_function"
    if left["capability"] != right["capability"]:
        return False, "different_operation"
    if left["desired_state"] != right["desired_state"]:
        return False, "opposite_or_different_state"
    if left["device"] == right["device"]:
        return False, "same_device_alias_not_new_behavior"
    return True, "same_function_operation_and_desired_state_across_catalogue_devices"


def build_action_mappings(semantics, embeddings, *, min_cosine=0.65, top_k=3):
    if isinstance(min_cosine, bool) or not isinstance(min_cosine, (int, float)) or not math.isfinite(min_cosine) or not 0 <= min_cosine <= 1:
        raise ValueError("mapping cosine threshold must be in [0, 1]")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("mapping top_k must be a positive integer")
    import numpy as np

    names, labels = embeddings["action_names"], semantics["records"]
    matrix = np.asarray(embeddings["vectors"], dtype=np.float64)
    cosines = np.clip(matrix @ matrix.T, -1.0, 1.0)
    eligible, reasons, near_misses = defaultdict(list), Counter(), []
    mask_pairs = 0
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if i == j:
                continue
            allowed, reason = compatibility(labels[left], labels[right])
            cosine = float(cosines[i, j])
            if not allowed:
                reasons[reason] += 1
                if cosine >= min_cosine:
                    near_misses.append({"source_action": left, "mapped_action": right,
                                        "cosine": cosine, "reason": reason})
                continue
            mask_pairs += 1
            if cosine < min_cosine:
                reasons["below_cosine_threshold"] += 1
                continue
            eligible[left].append({
                "source_action": left, "mapped_action": right, "cosine": cosine,
                "compatibility": reason, "environment_preference_evidence": "unavailable",
                "function": labels[left]["function"],
            })
    selected = []
    rows = {}
    for name in names:
        neighbors = sorted(eligible[name], key=lambda m: (-m["cosine"], m["mapped_action"]))[:top_k]
        denominator = 1.0 + sum(max(0.0, m["cosine"]) for m in neighbors)
        row = [{"action": name, "weight": 1.0 / denominator, "identity": True, "cosine": 1.0}]
        for neighbor in neighbors:
            neighbor["mapping_weight"] = max(0.0, neighbor["cosine"]) / denominator
            selected.append(neighbor)
            row.append({"action": neighbor["mapped_action"], "weight": neighbor["mapping_weight"],
                        "identity": False, "cosine": neighbor["cosine"]})
        rows[name] = row
    return {
        "mappings": selected, "sparse_rows": rows,
        "statistics": {
            "dictionary_actions": len(names), "directed_pairs_tested": len(names) * (len(names) - 1),
            "compatible_pairs_before_similarity": mask_pairs,
            "accepted_directed_mappings": len(selected),
            "mapped_source_action_count": len({m["source_action"] for m in selected}),
            "function_families": dict(Counter(m["function"] for m in selected)),
            "rejected_pair_reasons": dict(sorted(reasons.items())),
            "target_environment_specific_mappings": 0,
        },
        "high_similarity_rejected_examples": sorted(near_misses, key=lambda m: (-m["cosine"], m["source_action"], m["mapped_action"]))[:20],
        "config": {"min_cosine": min_cosine, "top_k": top_k,
                   "matrix_P": "row-normalized positive cosine after compatibility mask, plus identity weight 1",
                   "environment_gate": "catalogue capability only; target preference is unobserved"},
    }


def _observed_counts(raw_sequences, semantics, specifications):
    gss_sequences, _ = _decode_events(raw_sequences, semantics)
    gcad_sequences = _decode_gcad_events(raw_sequences, semantics)
    counts = {}
    for unit, lag in specifications:
        result = Counter()
        sequences = [sequence[:10] for sequence in gss_sequences] if unit == "raw_event_positions" else gcad_sequences
        for sequence in sequences:
            for left, right in zip(sequence, sequence[lag:]):
                if left is not None and right is not None:
                    result[(left["action"], right["action"])] += 1
        counts[(unit, lag)] = result
    return counts


def transfer_relations(source_artifact, mappings, raw_sequences, *,
                       transfer_sources="both", max_changed_endpoints=1,
                       max_candidates=12, max_per_target=4):
    if source_artifact.get("mode") != "typed":
        raise ValueError("transport requires typed source evidence before environment-policy deferral")
    if transfer_sources not in {"gss", "gcad", "both"}:
        raise ValueError("transfer sources must be gss, gcad or both")
    if isinstance(max_changed_endpoints, bool) or max_changed_endpoints not in {1, 2}:
        raise ValueError("max_changed_endpoints must be 1 or 2")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (max_candidates, max_per_target)):
        raise ValueError("candidate limits must be positive integers")
    source_records = source_artifact["relationships"] + source_artifact["unselected_relationships"]
    # Transport uses typed source evidence. A deferred semantic policy is not an
    # observation; it cannot be converted into a high-confidence source edge.
    kinds = {"observed_transition", "predictive_lag"}
    if transfer_sources != "both":
        kinds = {"observed_transition" if transfer_sources == "gss" else "predictive_lag"}
    edges = [(r, e) for r in source_records for e in r["evidence"] if e["relation_type"] in kinds]
    max_strength = max((e["raw_strength"] for _, e in edges if e["relation_type"] == "predictive_lag"), default=1.0)
    observed = _observed_counts(raw_sequences, source_artifact["action_semantics"],
                                {(e["lag_unit"], e["lag"]) for _, e in edges})
    candidates = {}
    mapped_source_edges = set()
    for parent, evidence in edges:
        a, b = parent["source_action"], parent["target_action"]
        for left in mappings["sparse_rows"][a]:
            for right in mappings["sparse_rows"][b]:
                changed = int(not left["identity"]) + int(not right["identity"])
                if not 1 <= changed <= max_changed_endpoints:
                    continue
                new_a, new_b = left["action"], right["action"]
                if new_a == new_b:
                    continue  # Avoid collapsing a two-action relation into a self loop.
                kind, lag, unit = evidence["relation_type"], evidence["lag"], evidence["lag_unit"]
                key = (kind, new_a, new_b, lag, unit)
                match_count = observed[(unit, lag)][(new_a, new_b)]
                candidate = candidates.setdefault(key, {
                    "relation_type": kind, "source_action": new_a, "target_action": new_b,
                    "source_lag_prior": lag, "lag_unit": unit,
                    "lag_validity_in_target": "unverified; do not enforce exact target lag",
                    "source_match_count": match_count, "target_observation_count": None,
                    "novel_against_source_events": match_count == 0,
                    "environment_preference_evidence": "unavailable",
                    "policy": "optional functional-transfer hypothesis; discard if inconsistent with target context",
                    "contributions": [], "ranking_score": 0.0,
                })
                q_src = (evidence["conditional_probability"] if kind == "observed_transition"
                         else evidence["raw_strength"] / max_strength * evidence["stability"])
                q_sem = left["weight"] * right["weight"]
                # Both endpoints passed the catalogue/compatibility gates.
                # q_env=1 is a gate, NOT learned environmental suitability.
                score = q_src * q_sem
                candidate["ranking_score"] += score
                candidate["contributions"].append({
                    "parent_source_action": a, "parent_target_action": b,
                    "source_evidence": evidence, "changed_endpoints": changed,
                    "q_source": q_src, "q_semantic_mapping": q_sem,
                    "q_environment_catalogue_gate": 1.0, "score": score,
                    "endpoint_mapping": {"source": left, "target": right},
                })
                mapped_source_edges.add((a, b, kind, lag, unit))
    all_candidates = sorted(candidates.values(), key=lambda c: (
        c["relation_type"], -c["ranking_score"], c["source_action"], c["target_action"], c["source_lag_prior"],
    ))
    new_candidates = [c for c in all_candidates if c["novel_against_source_events"]]
    # Keep independent rankings by evidence type; do not add C and G scores.
    pools = [[c for c in new_candidates if c["relation_type"] == kind]
             for kind in ("predictive_lag", "observed_transition")]
    positions, target_counts, selected = [0, 0], Counter(), []
    while len(selected) < max_candidates:
        added = False
        for index, pool in enumerate(pools):
            while positions[index] < len(pool):
                candidate = pool[positions[index]]
                positions[index] += 1
                if target_counts[candidate["target_action"]] >= max_per_target:
                    continue
                target_counts[candidate["target_action"]] += 1
                selected.append(candidate)
                added = True
                break
            if len(selected) >= max_candidates:
                break
        if not added:
            break
    by_kind = {}
    for kind in ("observed_transition", "predictive_lag"):
        denominator = sum(e["relation_type"] == kind for _, e in edges)
        covered = sum(key[2] == kind for key in mapped_source_edges)
        by_kind[kind] = {"eligible_source_edges": denominator, "source_edges_with_mapping": covered,
                         "coverage": covered / denominator if denominator else 0.0,
                         "candidate_count": sum(c["relation_type"] == kind for c in all_candidates),
                         "novel_candidate_count": sum(c["relation_type"] == kind for c in new_candidates)}
    return {
        "version": TRANSFER_VERSION, "all_candidates": all_candidates,
        "selected_candidates": selected, "by_relation_type": by_kind,
        "config": {"transfer_sources": transfer_sources,
                   "max_changed_endpoints": max_changed_endpoints,
                   "max_candidates": max_candidates, "max_per_target": max_per_target},
        "statistics": {"all_candidate_count": len(all_candidates),
                       "novel_candidate_count": len(new_candidates),
                       "selected_count": len(selected),
                       "environment_supported_candidate_count": 0},
        "score_interpretation": "sparse P^T C P and P^T G P contributions, restricted by changed-endpoint limit; type-specific normalized source weights; scores are not calibrated probabilities",
        "limitations": ["No target observations or environment-specific preferences establish these transfers.",
                        "A source lag is a proposal, not an identified target lag.",
                        "An existing source match is not counted as novel generated relation information."],
    }
