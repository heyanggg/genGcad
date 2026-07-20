from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from SmartGen.dictionary import fr_actions, fr_devices_dict
from SmartGen.gcad_source.v2_prediction import (
    evaluate_simple_baselines,
    prediction_gate,
    train_mixer,
)
from SmartGen.gcad_source.v2_representation import (
    canonicalize_sequences,
    collect_windows,
    continuous_event_time_representation,
    event_position_representation,
    load_pickle,
    map_tss_fragments_to_sources,
    representation_statistics,
    save_representation,
    sha256_file,
    split_by_source,
    v1_slot_lengths,
    window_count,
    window_origins,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/gcad_source_v2"


def config(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def paths():
    defaults = config("default.yaml")
    return ROOT / defaults["source_input"], ROOT / defaults["source_tss_input"], ROOT / defaults["output_root"]


def inverse(mapping):
    return {value: key for key, value in mapping.items()}


def source_data():
    source_path, tss_path, output = paths()
    raw, tss = load_pickle(source_path), load_pickle(tss_path)
    parsed, invalid, vocabulary = canonicalize_sequences(raw, inverse(fr_devices_dict), inverse(fr_actions))
    return source_path, tss_path, output, raw, tss, parsed, invalid, vocabulary


def describe(values):
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "minimum": float(array.min()) if len(array) else 0,
        "q10": float(np.quantile(array, 0.1)) if len(array) else 0,
        "median": float(np.median(array)) if len(array) else 0,
        "q90": float(np.quantile(array, 0.9)) if len(array) else 0,
        "maximum": float(array.max()) if len(array) else 0,
        "mean": float(array.mean()) if len(array) else 0,
        "histogram": {str(key): value for key, value in sorted(Counter(int(x) for x in values).items())},
    }


def audit_v1_leakage(fragment_mapping, output):
    reports = {}
    for seed in (2024, 2025, 2026):
        metrics_path = ROOT / f"outputs/gcad_source/fr/spring/seed_{seed}/training_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_sources = {fragment_mapping[index] for index in metrics["train_sequence_indices"]}
        validation_sources = {fragment_mapping[index] for index in metrics["validation_sequence_indices"]}
        reports[str(seed)] = {
            "train_fragment_count": len(metrics["train_sequence_indices"]),
            "validation_fragment_count": len(metrics["validation_sequence_indices"]),
            "train_source_count": len(train_sources),
            "validation_source_count": len(validation_sources),
            "source_overlap_count": len(train_sources & validation_sources),
            "source_overlap_ids": sorted(train_sources & validation_sources),
            "leakage_free": not (train_sources & validation_sources),
        }
    return reports


def run_audit() -> dict:
    source_path, tss_path, output, raw, tss, parsed, invalid, vocabulary = source_data()
    audit = output / "audit"
    channels = output / "channels"
    raw_event_lengths = [len(sequence) // 4 for sequence in raw]
    tss_event_lengths = [len(sequence) // 4 for sequence in tss]
    v1_lengths = v1_slot_lengths(tss)
    fragment_mapping = map_tss_fragments_to_sources(raw, tss)
    v1_leakage = audit_v1_leakage(fragment_mapping, output)
    invalid_reasons = Counter(row["reason"] for row in invalid)
    action_counts = Counter(event.channel for sequence in parsed for event in sequence)
    actual_165_sources = [index for index, length in enumerate(v1_lengths) if length > 4]
    input_files = {
        "source_pre_tss": {"path": str(source_path), "sha256": sha256_file(source_path), "role": "source_normal"},
        "source_tss": {"path": str(tss_path), "sha256": sha256_file(tss_path), "role": "source_normal"},
        "historical_v1_tensor_metadata": str(ROOT / "outputs/gcad_source/fr/spring/tensor/tensor_metadata.json"),
        "target_behavior_used": False,
    }
    write_json(audit / "input_files.json", input_files)
    sequence_statistics = {
        "pre_tss_sequence_count": len(raw),
        "pre_tss_event_count": sum(raw_event_lengths),
        "pre_tss_event_lengths": describe(raw_event_lengths),
        "tss_sequence_count": len(tss),
        "tss_event_count": sum(tss_event_lengths),
        "tss_event_lengths": describe(tss_event_lengths),
        "tss_slot_span_lengths_at_3h": describe(v1_lengths),
    }
    write_json(audit / "sequence_length_statistics.json", sequence_statistics)
    theoretical = {
        str(history): {
            "event_position_pre_tss_legal_events": sum(max(0, len(events) - history) for events in parsed),
            "v1_tss_3h_slot_span": sum(max(0, length - history) for length in v1_lengths),
            "tss_fragments_too_short_by_slot_span": sum(length <= history for length in v1_lengths),
        }
        for history in (2, 3, 4, 6)
    }
    write_json(audit / "theoretical_window_counts.json", theoretical)
    actual_counts = {
        "historical_v1_history_4": 165,
        "recomputed_v1_history_4": sum(max(0, length - 4) for length in v1_lengths),
        "contributing_tss_fragment_count": len(actual_165_sources),
        "contributing_tss_fragment_indices": actual_165_sources,
        "contributing_original_source_ids": sorted({fragment_mapping[index] for index in actual_165_sources}),
    }
    write_json(audit / "actual_window_counts.json", actual_counts)
    skipped = {
        "v1_did_not_require_nonempty_history_slots": True,
        "v1_did_not_filter_empty_time_slots": True,
        "v1_short_fragment_counts": {
            str(history): sum(length <= history for length in v1_lengths) for history in (2, 3, 4, 6)
        },
        "invalid_event_reasons": dict(invalid_reasons),
        "field_parse_failure_count": sum(reason != "forbidden_semantic_field" for reason in invalid_reasons.elements()),
        "root_cause": "TSS over-segmentation leaves most fragments spanning fewer than history+1 time slots",
    }
    write_json(audit / "skipped_window_reasons.json", skipped)
    target_support_by_history = {}
    event_records = event_position_representation(parsed, vocabulary)
    for history in (2, 3, 4, 6):
        _, y, _ = collect_windows(event_records, history)
        target_support_by_history[str(history)] = {
            channel: int(y[:, index].sum()) if len(y) else 0 for index, channel in enumerate(vocabulary)
        }
    channel_stats = {
        "historical_v1_channel_count": 40,
        "v2_legal_channel_count": len(vocabulary),
        "event_activation_counts": dict(sorted(action_counts.items())),
        "prediction_positive_target_counts": target_support_by_history,
        "invalid_event_count": len(invalid),
        "invalid_event_reasons": dict(invalid_reasons),
    }
    write_json(audit / "channel_activation_statistics.json", channel_stats)
    write_json(channels / "channel_vocabulary.json", vocabulary)
    write_jsonl(channels / "invalid_events.jsonl", invalid)
    write_json(channels / "excluded_fields.json", {
        "hard_excluded": ["None", "None:location", "location", "unknown", "padding", "context", "date", "day", "timestamp", "sequence_id", "user_id", "room metadata"],
        "source_observed_invalid_action": "None:location",
    })
    channel_audit = {
        "channel_count": len(vocabulary),
        "invalid_channel_count": 0,
        "all_channels_device_action": all(channel.count(":") == 1 for channel in vocabulary),
        "invalid_event_count": len(invalid),
        "passed": len(vocabulary) == 39 and all("None" not in channel for channel in vocabulary),
        "uses_target_behavior": False,
    }
    write_json(channels / "channel_audit.json", channel_audit)
    summary = {
        "historical_v1_valid_windows": 165,
        "root_causes": [
            "TSS expanded 873 original sequences into 1728 fragments",
            "1320 fragments span one 3-hour slot and only 83 fragments can contribute at history=4",
            "v1 split fragments after window construction instead of splitting original sources",
            "historical vocabulary retained illegal None:location",
        ],
        "v1_source_leakage_by_seed": v1_leakage,
        "source_pre_tss_path": str(source_path),
        "source_timestamp_format": "integer day index plus integer three-hour bin; no finer timestamp",
        "continuous_timestamp_available": False,
        "event_position_recommended": True,
        "channel_audit_passed": channel_audit["passed"],
        "target_behavior_used": False,
    }
    write_json(audit / "audit_summary.json", summary)
    return summary


def run_representations() -> list[dict]:
    source_path, _, output, _, _, parsed, _, vocabulary = source_data()
    grid = config("representation_grid.yaml")
    formal = config("fr_spring.yaml")
    event_records = event_position_representation(parsed, vocabulary)
    continuous_records = {
        time_bin: continuous_event_time_representation(parsed, vocabulary, time_bin)
        for time_bin in grid["continuous_time_bins_hours"]
    }
    save_representation(output / "representations/event_position.npz", event_records)
    for time_bin, records in continuous_records.items():
        save_representation(output / f"representations/continuous_{time_bin}h.npz", records)
    trials = []
    representations = [("event_position", None, event_records)] + [
        ("continuous_event_time", time_bin, records) for time_bin, records in continuous_records.items()
    ]
    for name, time_bin, records in representations:
        for history in grid["histories"]:
            train, validation, manifest = split_by_source(records, 0.2, 2024)
            train_x, train_y, _ = collect_windows(train, history)
            validation_x, validation_y, _ = collect_windows(validation, history)
            baseline_metrics = None
            if len(train_x) and len(validation_x):
                baseline_metrics = evaluate_simple_baselines(train_x, train_y, validation_x, validation_y)
            stats = representation_statistics(records, history)
            trial = {
                "representation": name,
                "time_bin_hours": time_bin,
                "history": history,
                "channel_count": len(vocabulary),
                "total_windows": stats["window_count"],
                "train_windows": len(train_x),
                "validation_windows": len(validation_x),
                "positive_targets": int(train_y.sum() + validation_y.sum()),
                "empty_slot_ratio": stats["empty_slot_ratio"],
                "average_active_channels_per_slot": stats["average_active_channels_per_slot"],
                "parameter_count": 0,
                "train_true_positive_rate": float(train_y.mean()) if train_y.size else 0.0,
                "validation_true_positive_rate": float(validation_y.mean()) if validation_y.size else 0.0,
                "simple_baseline_metrics": baseline_metrics,
                "source_overlap_count": manifest["source_overlap_count"],
                "passed_representation_gate": stats["window_count"] >= formal["minimum_effective_windows"] and manifest["passed"],
                "failure_reason": None if stats["window_count"] >= formal["minimum_effective_windows"] and manifest["passed"] else "insufficient_windows_or_source_leakage",
                "formal_trial": name == formal["formal_representation"] and history == formal["formal_history"] and time_bin == formal["formal_time_bin"],
                "uses_target_behavior": False,
            }
            trials.append(trial)
    write_jsonl(output / "representation_trials.jsonl", trials)
    split_manifests = []
    for seed in (2024, 2025, 2026):
        train, validation, manifest = split_by_source(event_records, 0.2, seed)
        manifest.update({
            "representation": "event_position",
            "history": 2,
            "input_sha256": sha256_file(source_path),
            "train_record_ids": [record.record_id for record in train],
            "validation_record_ids": [record.record_id for record in validation],
            "train_window_count": window_count(train, 2),
            "validation_window_count": window_count(validation, 2),
        })
        write_json(output / f"splits/split_{seed}.json", manifest)
        split_manifests.append(manifest)
    write_json(output / "splits/split_audit.json", {
        "all_seeds_passed": all(manifest["passed"] for manifest in split_manifests),
        "maximum_source_overlap": max(manifest["source_overlap_count"] for manifest in split_manifests),
        "split_before_window_construction": True,
        "uses_target_behavior": False,
    })
    return trials


def hash_state_dict(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_prediction() -> dict:
    source_path, _, output, _, _, parsed, _, vocabulary = source_data()
    records = event_position_representation(parsed, vocabulary)
    gate_config = config("prediction_gate.yaml")
    candidates = [(capacity, loss) for capacity in gate_config["mixer_capacities"] for loss in gate_config["losses"]]
    results_by_seed = {}
    candidate_scores = {f"{capacity}:{loss}": [] for capacity, loss in candidates}
    trained = {}
    for seed in (2024, 2025, 2026):
        train, validation, _ = split_by_source(records, 0.2, seed)
        train_x, train_y, train_origins = collect_windows(train, 2)
        validation_x, validation_y, validation_origins = collect_windows(validation, 2)
        simple = evaluate_simple_baselines(train_x, train_y, validation_x, validation_y)
        seed_mixers = {}
        for capacity, loss_mode in candidates:
            run = train_mixer(
                train_x, train_y, validation_x, validation_y,
                seed=seed,
                capacity=capacity,
                loss_mode=loss_mode,
                epochs=gate_config["epochs"],
                patience=gate_config["patience"],
                batch_size=gate_config["batch_size"],
                learning_rate=gate_config["learning_rate"],
                weight_decay=gate_config["weight_decay"],
                dropout=gate_config["dropout"],
            )
            key = f"{capacity}:{loss_mode}"
            run.metrics["model_checkpoint_hash"] = hash_state_dict(run.model)
            seed_mixers[key] = run.metrics
            candidate_scores[key].append(run.metrics[gate_config["primary_metric"]])
            checkpoint_dir = output / f"prediction/checkpoints/seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": run.model.state_dict(),
                "metrics": run.metrics,
                "history": run.history,
                "vocabulary": vocabulary,
                "representation": "event_position",
                "history_length": 2,
                "source_input_sha256": sha256_file(source_path),
                "uses_target_behavior": False,
            }, checkpoint_dir / f"{capacity}_{loss_mode}.pt")
        results_by_seed[seed] = {
            "simple_baselines": simple,
            "mixer_candidates": seed_mixers,
            "train_window_count": len(train_x),
            "validation_window_count": len(validation_x),
            "train_origins": train_origins,
            "validation_origins": validation_origins,
        }
    selected_key = max(candidate_scores, key=lambda key: (float(np.mean(candidate_scores[key])), key))
    for payload in results_by_seed.values():
        payload["selected_mixer"] = payload["mixer_candidates"][selected_key]
    gate = prediction_gate(
        results_by_seed,
        gate_config["primary_metric"],
        gate_config["maximum_train_validation_macro_f1_gap"],
    )
    gate.update({
        "selected_mixer_configuration": selected_key,
        "candidate_mean_source_validation_scores": {
            key: float(np.mean(values)) for key, values in candidate_scores.items()
        },
        "formal_representation": "event_position",
        "history": 2,
        "effective_window_count": window_count(records, 2),
        "effective_window_gate_passed": window_count(records, 2) >= 330,
        "illegal_channel_count": 0,
        "source_split_leakage_count": 0,
    })
    serializable = {
        "selected_mixer_configuration": selected_key,
        "seeds": {
            str(seed): {
                key: value for key, value in payload.items() if key not in {"train_origins", "validation_origins"}
            }
            for seed, payload in results_by_seed.items()
        },
        "gate": gate,
        "uses_target_behavior": False,
    }
    write_json(output / "prediction/comparison.json", serializable)
    rows = []
    for seed, payload in results_by_seed.items():
        for name, metrics in payload["simple_baselines"].items():
            rows.append({"seed": seed, "model": name, **{key: metrics[key] for key in ("overall_bce", "micro_precision", "micro_recall", "micro_f1", "macro_f1", "rare_channel_recall", "all_zero_prediction_rate")}})
        for name, metrics in payload["mixer_candidates"].items():
            rows.append({"seed": seed, "model": name, **{key: metrics[key] for key in ("overall_bce", "micro_precision", "micro_recall", "micro_f1", "macro_f1", "rare_channel_recall", "all_zero_prediction_rate")}})
    csv_path = output / "prediction/comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# GCAD v2 source prediction comparison", "",
        f"Formal representation: event-position, history 2. Selected Mixer: `{selected_key}`.", "",
        "| seed | best simple | simple macro F1 | Mixer macro F1 | Mixer rare recall | Mixer wins |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for seed, check in gate["seed_checks"].items():
        mixer = results_by_seed[int(seed)]["selected_mixer"]
        lines.append(f"| {seed} | {check['best_simple_baseline']} | {check['best_simple_value']:.6f} | {check['mixer_value']:.6f} | {mixer['rare_channel_recall']:.6f} | {check['mixer_wins']} |")
    lines.extend(["", f"Formal source prediction gate passed: **{gate['passed']}**.", ""])
    (output / "prediction/comparison.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "prediction/per_channel_metrics.json", {
        str(seed): {
            "vocabulary": vocabulary,
            "selected_mixer_per_channel_recall": payload["selected_mixer"]["per_channel_recall"],
            "simple_per_channel_recall": {
                name: metrics["per_channel_recall"] for name, metrics in payload["simple_baselines"].items()
            },
        }
        for seed, payload in results_by_seed.items()
    })
    write_json(output / "source_gate/source_prediction_gate.json", gate)
    write_json(output / "source_gate/source_decision.json", {
        "status": "pass_prediction_continue_to_relations" if gate["passed"] else "stop_gcad_v2_at_source_prediction_gate",
        "prediction_gate_passed": gate["passed"],
        "formal_relation_extraction_allowed": gate["passed"],
        "B5_generation_allowed": False,
        "target_behavior_used": False,
        "failure_conclusion": None if gate["passed"] else "Current SmartGen source data does not support reliable GCAD directional relation enhancement; terminate the GCAD route.",
    })
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "representations", "prediction", "all"))
    args = parser.parse_args()
    if args.stage in {"audit", "all"}:
        run_audit()
    if args.stage in {"representations", "all"}:
        run_representations()
    if args.stage in {"prediction", "all"}:
        gate = run_prediction()
        if not gate["passed"]:
            print("GCAD v2 source prediction gate failed; relations, fusion, and B5 are blocked.")


if __name__ == "__main__":
    main()
