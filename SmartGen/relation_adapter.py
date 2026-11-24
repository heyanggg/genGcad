"""Typed source evidence and conservative environment policies for generation."""
from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy


ADAPTER_VERSION = "typed-environment-policy-v1"
ENVIRONMENT_PAIRS = {"winter": "spring", "daytime": "night", "single": "multiple"}
NIGHT_HOUR_BINS = [6, 7, 0, 1]  # Explicit night-active task definition, not learned.


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _decode_events(raw_sequences, semantics):
    """Keep invalid events as boundaries; never silently bridge a GSS pair."""
    records = semantics["records"]
    by_id = {record["action_id"]: name for name, record in records.items()}
    result, excluded_reasons = [], Counter()
    for raw in raw_sequences:
        if len(raw) % 4:
            raise ValueError("source sequence is not a complete series of quadruplets")
        events = []
        for offset in range(0, len(raw), 4):
            day, hour, device_id, action_id = raw[offset:offset + 4]
            action = by_id.get(action_id)
            record = records.get(action, {})
            valid = (record.get("valid_device_action", False)
                     and device_id == record.get("device_id")
                     and isinstance(day, int) and not isinstance(day, bool) and 0 <= day <= 6
                     and isinstance(hour, int) and not isinstance(hour, bool) and 0 <= hour <= 7)
            events.append({"action": action, "day": day, "hour": hour} if valid else None)
            if not valid:
                if not record.get("valid_device_action", False):
                    reason = "non_device_or_unknown_action"
                elif device_id != record.get("device_id"):
                    reason = "device_action_id_mismatch"
                else:
                    reason = "invalid_day_or_hour_bin"
                excluded_reasons[reason] += 1
        result.append(events)
    return result, excluded_reasons


def _decode_gcad_events(raw_sequences, semantics):
    """Match upstream GCAD positions: time fields do not change its indexing."""
    records = semantics["records"]
    by_id = {record["action_id"]: record for record in records.values()}
    names = {record["action_id"]: name for name, record in records.items()}
    decoded = []
    for raw in raw_sequences:
        sequence = []
        for offset in range(0, len(raw), 4):
            record = by_id.get(raw[offset + 3], {})
            if record.get("valid_device_action") and raw[offset + 2] == record["device_id"]:
                hour = raw[offset + 1]
                sequence.append({"action": names[raw[offset + 3]],
                                 "hour": hour if isinstance(hour, int) and not isinstance(hour, bool) and 0 <= hour <= 7 else None})
        decoded.append(sequence)
    return decoded


def _adapt_policy(record, semantics, source_environment, target_environment, mode):
    if mode == "typed":
        return {"decision": "unadapted_reference", "target_validity": "unverified",
                "reason": "source evidence organized by type; no environment adaptation"}
    labels = [semantics["records"][record[key]] for key in ("source_action", "target_action")]
    if any(label["semantic_status"] == "unresolved" for label in labels):
        return {"decision": "defer", "target_validity": "unknown",
                "reason": "function or command state unresolved by the frozen label rules"}
    policy = {"decision": "soft_reference", "target_validity": "unverified",
              "reason": "no explicit task conflict identified; target persistence is not established"}
    if target_environment == "night":
        policy.update(
            decision="retime_preference",
            reason="source hour preferences do not specify target hours; retain order only as a soft hypothesis",
            target_preferred_hour_bins=NIGHT_HOUR_BINS,
            time_resolution_hours=3,
            policy_source="existing SmartGen night-active task definition",
        )
    elif target_environment == "spring" and any(
        "season_and_unobserved_temperature" in label["potential_environment_dependence"] for label in labels
    ):
        policy.update(
            decision="conditional_reference",
            reason="season changed but temperature, setpoints, and user comfort are unobserved",
            instruction="do not require or prohibit climate actions solely because the target is spring",
        )
    elif target_environment == "multiple":
        policy.update(
            decision="conditional_reference",
            reason="events lack resident identity; per-person habits and concurrent use cannot be inferred",
            instruction="do not duplicate this relationship per resident or infer person-specific ownership",
        )
    return policy


def adapt_relations(gss, gcad, semantics, raw_sequences, *, dataset,
                    source_environment, target_environment, source_sha256,
                    mode="adapted", min_support=3, max_relationships=12,
                    max_per_target=4):
    if mode not in {"typed", "adapted"}:
        raise ValueError("relation mode must be typed or adapted")
    if ENVIRONMENT_PAIRS.get(source_environment) != target_environment:
        raise ValueError("unsupported environment transition")
    if not all(_positive_int(v) for v in (min_support, max_relationships, max_per_target)):
        raise ValueError("relation support and selection limits must be positive integers")
    semantics = deepcopy(semantics)
    records = semantics["records"]
    events, excluded_reasons = _decode_events(raw_sequences, semantics)
    # Match upstream GSS exactly: at most the first ten raw event positions.
    pair_counts, raw_pair_counts, outgoing = Counter(), Counter(), Counter()
    hour_counts, day_counts = {}, {}
    for raw, sequence in zip(raw_sequences, events):
        for offset in range(3, min(len(raw), 40) - 4, 4):
            outgoing[raw[offset]] += 1
            raw_pair_counts[(raw[offset], raw[offset + 4])] += 1
        for left, right in zip(sequence[:10], sequence[1:10]):
            if left is None or right is None:
                continue
            key = (left["action"], right["action"])
            pair_counts[key] += 1
            hour_counts.setdefault(key, Counter())[right["hour"]] += 1
            day_counts.setdefault(key, Counter())[right["day"]] += 1

    candidates, rejected = {}, []

    def reject(kind, source, target, reason):
        rejected.append({"relation_type": kind, "source_action": source,
                         "target_action": target, "reason": reason})

    def valid_action(action):
        return action in records and records[action]["valid_device_action"]

    def add(source, target, evidence):
        record = candidates.setdefault((source, target), {
            "relation_id": f"{source} -> {target}", "source_action": source,
            "target_action": target, "evidence": [],
        })
        record["evidence"].append(evidence)

    for source, item in sorted(gss.items()):
        for edge in item.get("transitions", []):
            target, count = edge.get("next_action"), edge.get("count")
            if not valid_action(source) or not valid_action(target):
                reject("gss", source, target, "invalid_or_non_device_action")
                continue
            raw_count = raw_pair_counts[(records[source]["action_id"], records[target]["action_id"])]
            if not _positive_int(count) or count != raw_count:
                raise ValueError(f"GSS count does not match current source training data: {source} -> {target}")
            if count != pair_counts[(source, target)]:
                reject("gss", source, target, "source_pair_contains_invalid_events")
                continue
            if count < min_support:
                reject("gss", source, target, "insufficient_source_support")
                continue
            total = outgoing[records[source]["action_id"]]
            add(source, target, {
                "relation_type": "observed_transition", "source": "GSS",
                "lag": 1, "lag_unit": "raw_event_positions",
                "count": count, "source_outgoing_count": total,
                "conditional_probability": count / total,
                "target_hour_counts": {str(k): v for k, v in sorted(hour_counts[(source, target)].items())},
                "target_weekday_counts": {str(k): v for k, v in sorted(day_counts[(source, target)].items())},
                "time_resolution_hours": 3, "time_scope": "source_environment_only",
            })

    gcad_status = gcad.get("status", "missing")
    gcad_edges = gcad.get("lagged_behavior_relations", [])
    if gcad_status == "ready":
        provenance = gcad.get("source", {})
        if provenance.get("sha256") != source_sha256 or provenance.get("dataset") != dataset:
            raise ValueError("GCAD artifact does not match current source training data and dataset")
        if gcad.get("lag_unit", "subsequent_behavior_positions") != "subsequent_behavior_positions":
            raise ValueError("unsupported GCAD lag unit")
        # Mirror GCAD's valid-device decoding, which excludes non-device actions.
        gcad_sequences = _decode_gcad_events(raw_sequences, semantics)
        seen = set()
        for edge in gcad_edges:
            source, target = edge.get("source_action"), edge.get("target_action")
            if not valid_action(source) or not valid_action(target):
                reject("gcad", source, target, "invalid_or_non_device_action")
                continue
            lag, support = edge.get("typical_lag"), edge.get("support")
            stable, total = edge.get("stable_seeds"), edge.get("total_seeds")
            strength = edge.get("raw_strength")
            if (not all(_positive_int(v) for v in (lag, support, stable, total))
                    or stable > total or not _finite_nonnegative(strength) or strength == 0):
                raise ValueError("malformed GCAD relationship evidence")
            fraction = gcad.get("config", {}).get("stable_seed_fraction", 2 / 3)
            if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError("invalid GCAD stability configuration")
            if support < min_support or stable < math.ceil(total * fraction):
                reject("gcad", source, target, "insufficient_source_support_or_seed_stability")
                continue
            key = (source, target, lag)
            if key in seen:
                continue
            seen.add(key)
            target_hours = Counter()
            for sequence in gcad_sequences:
                for left, right in zip(sequence, sequence[lag:]):
                    if left["action"] == source and right["action"] == target and right["hour"] is not None:
                        target_hours[right["hour"]] += 1
            add(source, target, {
                "relation_type": "predictive_lag", "source": "GCAD-derived",
                "lag": lag, "lag_unit": "valid_device_behavior_positions",
                "support": support, "support_scope": "upstream held-out gradient evidence; median across seeds",
                "raw_strength": strength, "stable_seeds": stable, "total_seeds": total,
                "stability": stable / total, "cross_environment_stability": "not_established",
                "target_hour_counts": {str(k): v for k, v in sorted(target_hours.items())},
                "hour_count_scope": "descriptive source matches; not GCAD support",
                "time_resolution_hours": 3, "time_scope": "source_environment_only",
            })

    eligible, deferred = [], []
    for record in candidates.values():
        record["evidence"].sort(key=lambda e: (e["relation_type"], e["lag"]))
        record["policy"] = _adapt_policy(record, semantics, source_environment, target_environment, mode)
        (deferred if record["policy"]["decision"] == "defer" else eligible).append(record)

    def gcad_rank(record):
        evidence = [e for e in record["evidence"] if e["relation_type"] == "predictive_lag"]
        return min((-e["stability"], -e["raw_strength"], -e["support"], record["relation_id"]) for e in evidence)

    def gss_rank(record):
        e = next(e for e in record["evidence"] if e["relation_type"] == "observed_transition")
        return (-e["count"], -e["conditional_probability"], record["relation_id"])

    pools = [sorted([r for r in eligible if any(e["relation_type"] == "predictive_lag" for e in r["evidence"])], key=gcad_rank),
             sorted([r for r in eligible if any(e["relation_type"] == "observed_transition" for e in r["evidence"])], key=gss_rank)]
    selected, selected_ids, target_counts = [], set(), Counter()
    positions = [0, 0]
    while len(selected) < max_relationships:
        added = False
        for index, pool in enumerate(pools):
            while positions[index] < len(pool):
                record = pool[positions[index]]
                positions[index] += 1
                if record["relation_id"] in selected_ids or target_counts[record["target_action"]] >= max_per_target:
                    continue
                selected.append(record)
                selected_ids.add(record["relation_id"])
                target_counts[record["target_action"]] += 1
                added = True
                break
            if len(selected) >= max_relationships:
                break
        if not added:
            break
    unselected = [r for r in eligible if r["relation_id"] not in selected_ids]
    return {
        "schema_version": 1, "adapter_version": ADAPTER_VERSION, "mode": mode,
        "dataset": dataset, "source_environment": source_environment,
        "target_environment": target_environment,
        "status": "ready" if selected else "empty",
        "relationships": selected, "deferred_relationships": sorted(deferred, key=lambda r: r["relation_id"]),
        "unselected_relationships": sorted(unselected, key=lambda r: r["relation_id"]),
        "rejected_evidence": rejected,
        "selection": {"method": "round_robin_gcad_and_gss_within_type_ranking",
                      "max_relationships": max_relationships, "max_per_target": max_per_target,
                      "min_support": min_support, "candidate_count": len(candidates),
                      "selected_count": len(selected), "deferred_count": len(deferred),
                      "unselected_count": len(unselected), "rejected_evidence_count": len(rejected)},
        "source_audit": {"sequence_count": len(raw_sequences),
                         "invalid_event_count": sum(excluded_reasons.values()),
                         "excluded_event_reasons": dict(sorted(excluded_reasons.items())),
                         "gss_event_limit_per_sequence": 10, "gcad_status": gcad_status,
                         "gcad_disabled_reason": gcad.get("disabled_reason")},
        "semantic_substitution_enabled": False,
    }
