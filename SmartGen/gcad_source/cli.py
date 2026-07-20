from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from SmartGen import dictionary
from SmartGen.generation_backends.codex_file import CodexFileBackend

from .asymmetric_filter import save_asymmetric_relation
from .config import GCADConfig
from .data_roles import DataRole, RoleBoundPath
from .event_tensorizer import SourceEventTensorizer, load_tensorized
from .gradient_relation import extract_gradient_relation
from .gss_fusion import fuse_gss_files
from .prediction_baselines import compare_baselines
from .sequence_ranking import rank_sequences
from .stability_filter import build_stable_relation
from .trainer import load_checkpoint, train_model


def dataset_mappings(dataset: str):
    devices = getattr(dictionary, f"{dataset}_devices_dict")
    actions = getattr(dictionary, f"{dataset}_actions")
    return devices, actions


def target_metadata(dataset: str) -> dict[str, list[str]]:
    _, actions = dataset_mappings(dataset)
    result: dict[str, list[str]] = {}
    for channel in sorted(actions):
        device, action = channel.split(":", 1)
        result.setdefault(device, []).append(action)
    return result


def command_tensorize(args):
    config = GCADConfig.from_yaml(args.config)
    devices, actions = dataset_mappings(args.dataset)
    tensorizer = SourceEventTensorizer(
        config.channel_mode,
        config.time_bin,
        config.occurrence_mode,
        config.history_length,
        {value: key for key, value in devices.items()},
        {value: key for key, value in actions.items()},
        args.unknown_policy,
    )
    result = tensorizer.tensorize_pickle(
        RoleBoundPath.build(args.input, DataRole.SOURCE_NORMAL), args.output
    )
    print(json.dumps(result.metadata, indent=2))


def command_train(args):
    config = GCADConfig.from_yaml(args.config)
    config.device = args.device or config.device
    sequences, _, _ = load_tensorized(args.tensor_dir)
    result = train_model(sequences, config, args.output, args.seed)
    comparison = compare_baselines(
        result.model,
        result.train_sequences,
        result.validation_sequences,
        config.history_length,
        args.output,
        args.ngram_order,
        config.device,
    )
    print(json.dumps({"training": result.metrics, "baselines": comparison}, indent=2))


def command_extract(args):
    sequences, vocabulary, metadata = load_tensorized(args.tensor_dir)
    model, payload = load_checkpoint(args.checkpoint, args.device)
    result = extract_gradient_relation(
        model,
        sequences,
        payload["config"]["history_length"],
        args.output,
        vocabulary,
        args.batch_size,
        args.sample_aggregation,
        args.lag_aggregation,
        args.device,
    )
    filtered = save_asymmetric_relation(
        result["raw_relation"], vocabulary, args.output, args.edge_threshold, args.top_k, result["primary_lag"]
    )
    print(json.dumps(filtered["report"], indent=2))


def command_stable(args):
    vocabulary = json.loads(Path(args.vocabulary).read_text(encoding="utf-8"))
    matrices = [np.load(path) for path in args.matrices]
    primary_lags = [np.load(path) for path in args.primary_lags] if args.primary_lags else None
    result = build_stable_relation(
        matrices,
        vocabulary,
        primary_lags,
        args.occurrence_threshold,
        args.direction_consistency_threshold,
        args.stability_threshold,
        args.seeds,
        args.split_ids,
        args.output,
    )
    print(json.dumps(result["payload"], indent=2))


def command_fuse(args):
    fused, report = fuse_gss_files(
        args.original_gss, args.stable_relation, args.target_metadata, args.output, args.alpha, args.mode
    )
    print(json.dumps(report, indent=2))


def command_rank(args):
    with Path(args.sequences).open("rb") as handle:
        sequences = pickle.load(handle)
    stable = json.loads(Path(args.stable_relation).read_text(encoding="utf-8"))
    _, actions = dataset_mappings(args.dataset)
    result = rank_sequences(sequences, stable, {value: key for key, value in actions.items()}, args.ranking_weight)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"sequence_count": result["sequence_count"], "output": args.output}, indent=2))


def command_generate_export(args):
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    metadata = json.loads(Path(args.target_metadata).read_text(encoding="utf-8"))
    requests = CodexFileBackend().export(
        args.output,
        experiment_id=args.experiment_id,
        dataset=args.dataset,
        context=args.context,
        method=args.method,
        replicate=args.replicate,
        prompt=prompt,
        requested_sequence_count=args.count,
        batch_size=args.batch_size,
        source_sequence_count=args.source_sequence_count,
        target_context_description=json.loads(args.context_description),
        target_static_device_metadata=metadata,
        original_gss_path=args.original_gss,
        fused_gss_path=args.fused_gss,
        stable_relation_path=args.stable_relation,
    )
    print(json.dumps({"request_count": len(requests), "sequence_count": args.count}, indent=2))


def command_generate_validate(args):
    print(json.dumps(CodexFileBackend().validate(args.directory), indent=2))


def command_generate_convert(args):
    devices, actions = dataset_mappings(args.dataset)
    path = CodexFileBackend().convert(
        args.directory,
        day_mapping=dictionary.dayofweek_dict,
        hour_mapping=dictionary.hour_dict,
        device_mapping=devices,
        action_mapping=actions,
    )
    print(path)


def command_continue(args):
    from SmartGen.security_check import security_check_file

    directory = Path(args.directory)
    tof_path, report = security_check_file(
        directory / "generated_sequences.pkl", directory / "tof", args.dataset, args.context, args.tof_epochs
    )
    if args.stable_relation:
        with tof_path.open("rb") as handle:
            sequences = pickle.load(handle)
        stable = json.loads(Path(args.stable_relation).read_text(encoding="utf-8"))
        _, actions = dataset_mappings(args.dataset)
        ranking = rank_sequences(sequences, stable, {value: key for key, value in actions.items()}, args.ranking_weight)
        (directory / "tof" / "sequence_ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="source-gcad")
    sub = root.add_subparsers(dest="command", required=True)
    item = sub.add_parser("tensorize")
    item.add_argument("--input", required=True); item.add_argument("--output", required=True)
    item.add_argument("--dataset", required=True, choices=["fr", "sp", "us"])
    item.add_argument("--config", required=True); item.add_argument("--unknown-policy", default="error")
    item.set_defaults(function=command_tensorize)
    item = sub.add_parser("train")
    item.add_argument("--tensor-dir", required=True); item.add_argument("--output", required=True)
    item.add_argument("--config", required=True); item.add_argument("--seed", type=int, required=True)
    item.add_argument("--device"); item.add_argument("--ngram-order", type=int, default=2)
    item.set_defaults(function=command_train)
    item = sub.add_parser("extract-relations")
    item.add_argument("--tensor-dir", required=True); item.add_argument("--checkpoint", required=True)
    item.add_argument("--output", required=True); item.add_argument("--device", default="cpu")
    item.add_argument("--batch-size", type=int, default=64); item.add_argument("--sample-aggregation", default="mean")
    item.add_argument("--lag-aggregation", default="max"); item.add_argument("--edge-threshold", type=float, default=0.01)
    item.add_argument("--top-k", type=int, default=5); item.set_defaults(function=command_extract)
    item = sub.add_parser("build-stable-relation")
    item.add_argument("--matrices", nargs="+", required=True); item.add_argument("--primary-lags", nargs="*")
    item.add_argument("--vocabulary", required=True); item.add_argument("--output", required=True)
    item.add_argument("--occurrence-threshold", type=float, default=0.5)
    item.add_argument("--direction-consistency-threshold", type=float, default=0.5)
    item.add_argument("--stability-threshold", type=float, default=0.25)
    item.add_argument("--seeds", nargs="*", type=int); item.add_argument("--split-ids", nargs="*")
    item.set_defaults(function=command_stable)
    item = sub.add_parser("fuse-gss")
    item.add_argument("--original-gss", required=True); item.add_argument("--stable-relation", required=True)
    item.add_argument("--target-metadata", required=True); item.add_argument("--output", required=True)
    item.add_argument("--alpha", type=float, default=0.2); item.add_argument("--mode", default="rerank_existing")
    item.set_defaults(function=command_fuse)
    item = sub.add_parser("rank-sequences")
    item.add_argument("--sequences", required=True); item.add_argument("--stable-relation", required=True)
    item.add_argument("--dataset", required=True); item.add_argument("--output", required=True)
    item.add_argument("--ranking-weight", type=float, default=1.0); item.set_defaults(function=command_rank)
    item = sub.add_parser("export")
    item.add_argument("--output", required=True); item.add_argument("--experiment-id", required=True)
    item.add_argument("--dataset", required=True); item.add_argument("--context", required=True); item.add_argument("--method", required=True)
    item.add_argument("--replicate", type=int, default=1); item.add_argument("--prompt", required=True)
    item.add_argument("--count", type=int, required=True); item.add_argument("--batch-size", type=int, default=20)
    item.add_argument("--source-sequence-count", type=int, required=True); item.add_argument("--context-description", required=True)
    item.add_argument("--target-metadata", required=True); item.add_argument("--original-gss", required=True)
    item.add_argument("--fused-gss"); item.add_argument("--stable-relation"); item.set_defaults(function=command_generate_export)
    item = sub.add_parser("validate"); item.add_argument("--directory", required=True); item.set_defaults(function=command_generate_validate)
    item = sub.add_parser("convert"); item.add_argument("--directory", required=True); item.add_argument("--dataset", required=True); item.set_defaults(function=command_generate_convert)
    item = sub.add_parser("continue-pipeline")
    item.add_argument("--directory", required=True); item.add_argument("--dataset", required=True); item.add_argument("--context", required=True)
    item.add_argument("--tof-epochs", type=int, default=10); item.add_argument("--stable-relation")
    item.add_argument("--ranking-weight", type=float, default=1.0); item.set_defaults(function=command_continue)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

