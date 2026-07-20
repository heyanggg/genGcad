from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from SmartGen import dictionary

from .response_loader import load_jsonl


def normalized_ngram_entropy(sequences: list[list[str]], order: int) -> tuple[int, float]:
    counts = Counter(
        tuple(sequence[index:index + order])
        for sequence in sequences
        for index in range(len(sequence) - order + 1)
    )
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return len(counts), 0.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return len(counts), entropy / math.log(len(counts))


def effective_count(items) -> float:
    counts = np.asarray(list(Counter(items).values()), dtype=float)
    if not len(counts):
        return 0.0
    probabilities = counts / counts.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def sequence_metrics(action_sequences: list[list[str]]) -> dict:
    templates = [tuple(sequence) for sequence in action_sequences]
    counts = Counter(templates)
    lengths = [len(sequence) for sequence in action_sequences]
    actions = Counter(action for sequence in action_sequences for action in sequence)
    bigram_count, bigram_entropy = normalized_ngram_entropy(action_sequences, 2)
    trigram_count, trigram_entropy = normalized_ngram_entropy(action_sequences, 3)
    total = len(templates) or 1
    top = [count / total for _, count in counts.most_common(5)]
    opening = Counter(tuple(sequence[:2]) for sequence in action_sequences)
    ending = Counter(tuple(sequence[-2:]) for sequence in action_sequences)
    return {
        "sequence_count": len(templates),
        "length": {
            "min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths),
            "distribution": {str(key): value for key, value in sorted(Counter(lengths).items())},
            "distinct_count": len(set(lengths)),
        },
        "unique_sequence_count": len(counts),
        "unique_sequence_ratio": len(counts) / total,
        "exact_duplicate_count": len(templates) - len(counts),
        "unique_unigram_count": len(actions),
        "unique_bigram_count": bigram_count,
        "unique_trigram_count": trigram_count,
        "normalized_bigram_entropy": bigram_entropy,
        "normalized_trigram_entropy": trigram_entropy,
        "top_1_template_share": top[0] if top else 0.0,
        "top_5_template_share": sum(top),
        "effective_sequence_count": effective_count(templates),
        "opening_template_count": len(opening),
        "ending_template_count": len(ending),
        "top_opening_share": opening.most_common(1)[0][1] / total if opening else 0.0,
        "top_ending_share": ending.most_common(1)[0][1] / total if ending else 0.0,
        "action_frequency": dict(actions.most_common()),
        "maximum_action_share": max(actions.values()) / sum(actions.values()) if actions else 0.0,
    }


def _numeric_actions(sequences, dataset: str) -> list[list[str]]:
    mapping = getattr(dictionary, f"{dataset}_actions")
    inverse = {value: key for key, value in mapping.items()}
    return [[inverse[int(value)] for value in sequence[3::4]] for sequence in sequences]


def diagnose_grouped_generation(directory: str | Path, dataset: str, original_gss_path: str | Path) -> dict:
    directory = Path(directory)
    requests = load_jsonl(directory / "generation_requests.jsonl")
    responses = load_jsonl(directory / "generation_responses_validated.jsonl")
    request_map = {item["request_id"]: item for item in requests}
    generated_actions = []
    group_actions = defaultdict(list)
    group_templates = defaultdict(set)
    template_groups = defaultdict(set)
    for response in responses:
        request = request_map[response["request_id"]]
        group = request["group_id"]
        for sequence in response["sequences"]:
            actions = [
                event["action"] if ":" in event["action"] else f"{event['device']}:{event['action']}"
                for event in sequence["events"]
            ]
            generated_actions.append(actions)
            group_actions[group].append(actions)
            template = tuple(actions)
            group_templates[group].add(template)
            template_groups[template].add(group)
    source_numeric = []
    seen_source_paths = set()
    for request in requests:
        path = request["source_group_path"]
        if path not in seen_source_paths:
            source_numeric.extend(pickle.loads(Path(path).read_bytes()))
            seen_source_paths.add(path)
    source_actions = _numeric_actions(source_numeric, dataset)
    generated = sequence_metrics(generated_actions)
    source = sequence_metrics(source_actions)
    cross_group_duplicates = sum(1 for groups in template_groups.values() if len(groups) > 1)
    source_templates = {tuple(sequence) for sequence in source_actions}
    source_copy_count = sum(tuple(sequence) in source_templates for sequence in generated_actions)
    gss = json.loads(Path(original_gss_path).read_text(encoding="utf-8"))
    gss_pairs = {
        (source_action, item["next_action"])
        for source_action, record in gss.items()
        for item in record.get("transitions", [])
    }
    coverage = []
    for sequence in generated_actions:
        pairs = list(zip(sequence, sequence[1:]))
        coverage.append(sum(pair in gss_pairs for pair in pairs) / len(pairs) if pairs else 0.0)
    metadata = requests[0]["target_static_device_metadata"]
    legal_actions = {f"{device}:{action.split(':', 1)[-1]}" for device, actions in metadata.items() for action in actions}
    generated_unique_actions = {action for sequence in generated_actions for action in sequence}
    validation = json.loads((directory / "generation_validation_report.json").read_text(encoding="utf-8"))
    report = {
        "protocol": "SmartGen-compatible grouped Codex-file baseline v2",
        "generated": generated,
        "source_representatives": source,
        "group_count": len(group_actions),
        "group_metrics": {group: sequence_metrics(values) for group, values in sorted(group_actions.items())},
        "cross_group_duplicate_template_count": cross_group_duplicates,
        "cross_group_duplicate_ratio": cross_group_duplicates / max(1, generated["unique_sequence_count"]),
        "exact_source_representative_copy_count": source_copy_count,
        "device_coverage_count": len({action.split(":", 1)[0] for action in generated_unique_actions}),
        "action_vocabulary_coverage_count": len(generated_unique_actions),
        "action_vocabulary_coverage_ratio": len(generated_unique_actions & legal_actions) / max(1, len(legal_actions)),
        "mean_gss_transition_coverage": float(np.mean(coverage)),
        "per_sequence_gss_coverage": coverage,
        "illegal_device_action_count": validation["illegal_device_count"] + validation["illegal_action_count"],
        "uses_target_behavior": False,
    }
    source_bigram_floor = 0.8 * source["normalized_bigram_entropy"]
    source_trigram_floor = 0.8 * source["normalized_trigram_entropy"]
    checks = {
        "illegal_device_action_zero": report["illegal_device_action_count"] == 0,
        "exact_duplicates_zero": generated["exact_duplicate_count"] == 0,
        "unique_sequence_ratio": generated["unique_sequence_ratio"] >= min(0.98, source["unique_sequence_ratio"]),
        "lengths_vary": generated["length"]["distinct_count"] >= min(3, source["length"]["distinct_count"]),
        "bigram_entropy_source_relative": generated["normalized_bigram_entropy"] >= source_bigram_floor,
        "trigram_entropy_source_relative": generated["normalized_trigram_entropy"] >= source_trigram_floor,
        "top_template_share": generated["top_1_template_share"] <= max(0.1, source["top_1_template_share"]),
        "cross_group_duplicates_zero": cross_group_duplicates == 0,
        "source_exact_copies_zero": source_copy_count == 0,
        "maximum_action_share": generated["maximum_action_share"] <= max(0.25, 1.5 * source["maximum_action_share"]),
    }
    gate = {
        "stage": "pre_tof_generation_distribution",
        "passed": all(checks.values()),
        "checks": checks,
        "threshold_basis": "source representative and generated-internal statistics only",
        "uses_target_behavior": False,
    }
    report["generation_gate"] = gate
    (directory / "generation_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "generation_quality_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return report


def apply_reconstruction_gate(directory: str | Path, diagnostics_path: str | Path) -> dict:
    directory = Path(directory)
    diagnostics = json.loads(Path(diagnostics_path).read_text(encoding="utf-8"))
    validation = np.asarray(diagnostics["validation_losses"], dtype=float)
    checks = {
        "train_validation_exact_overlap_zero": diagnostics["split_integrity"]["exact_overlap_count"] == 0,
        "validation_not_numerical_zero": float(np.median(validation)) > 1e-4,
        "validation_has_dispersion": float(np.percentile(validation, 90) - np.percentile(validation, 10)) > 1e-4,
        "near_zero_fraction_below_half": float(np.mean(validation <= 1e-6)) < 0.5,
        "all_samples_covered": diagnostics["all_training_samples_covered_each_epoch"] is True,
    }
    gate = {
        "stage": "pre_target_reconstruction",
        "passed": all(checks.values()),
        "checks": checks,
        "threshold": diagnostics["threshold"],
        "validation_loss_summary": diagnostics["validation_loss_summary"],
        "threshold_basis": "fixed numerical-collapse checks; no target behavior or official target score used",
        "uses_target_behavior": False,
    }
    (directory / "reconstruction_quality_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate
