"""Measure semantic mapping and relation-transport feasibility before generation."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from SmartGen.experiment import atomic_write_json
from SmartGen.relation_preview import main as preview
from SmartGen.relation_transfer import build_action_mappings, load_embeddings, transfer_relations
from SmartGen.relation_stage import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.65)
    parser.add_argument("--mapping-top-k", type=int, default=3)
    parser.add_argument("--max-changed-endpoints", choices=[1, 2], type=int, default=1)
    args = parser.parse_args(argv)
    project, output = args.source_project.resolve(), args.output.resolve()
    if output == project or project in output.parents:
        parser.error("audit output must be outside the source project")
    summaries, dataset_mappings, embedding_hashes = [], {}, {}
    for dataset in ("fr", "sp", "us"):
        for source, target in (("winter", "spring"), ("daytime", "night"), ("single", "multiple")):
            directory = output / f"{dataset}_{source}_to_{target}"
            gcad_paths = sorted((project / "experiment_archive/formal").glob(
                f"{dataset}_{source}_to_{target}_*/artifacts/gcad/gcad_hints.json"
            ))
            if len(gcad_paths) != 1:
                raise ValueError(f"expected exactly one formal GCAD artifact for {dataset}/{source}")
            with contextlib.redirect_stdout(io.StringIO()):
                preview([
                    "--source-project", str(project), "--output", str(directory / "source_audit"),
                    "--dataset", dataset, "--ori-env", source, "--new-env", target,
                    "--relation-mode", "typed", "--gcad-artifact", str(gcad_paths[0]),
                ])
            artifact = json.loads((directory / "source_audit/relation_artifact.json").read_text(encoding="utf-8"))
            if dataset not in dataset_mappings:
                embedding_path = args.embedding_dir.resolve() / f"{dataset}_embeddings.json"
                embeddings = load_embeddings(embedding_path, artifact["action_semantics"], dataset)
                embedding_hashes[dataset] = file_sha256(embedding_path)
                mappings = build_action_mappings(
                    artifact["action_semantics"], embeddings,
                    min_cosine=args.min_cosine, top_k=args.mapping_top_k,
                )
                mappings["embedding_artifact_fingerprint"] = embeddings["artifact_fingerprint"]
                atomic_write_json(output / f"{dataset}_action_mappings.json", mappings)
                dataset_mappings[dataset] = mappings
            source_path = Path(artifact["inputs"]["source_training"]["path"])
            with source_path.open("rb") as handle:
                rows = pickle.load(handle)
            transfer = transfer_relations(
                artifact, dataset_mappings[dataset], rows,
                max_changed_endpoints=args.max_changed_endpoints,
            )
            transfer["source_artifact_fingerprint"] = artifact["artifact_fingerprint"]
            transfer["embedding_file_sha256"] = embedding_hashes[dataset]
            atomic_write_json(directory / "transfer_candidates.json", transfer)
            summary = {
                "dataset": dataset, "source_environment": source, "target_environment": target,
                "action_mapping_count": dataset_mappings[dataset]["statistics"]["accepted_directed_mappings"],
                "mapping_function_families": dataset_mappings[dataset]["statistics"]["function_families"],
                "by_relation_type": transfer["by_relation_type"],
                "statistics": transfer["statistics"],
                "source_sha256": artifact["inputs"]["source_training"]["sha256"],
                "source_sha256_unchanged": file_sha256(source_path) == artifact["inputs"]["source_training"]["sha256"],
                "gcad_status": artifact["source_audit"]["gcad_status"],
            }
            summaries.append(summary)
            print(f"{dataset} {source}->{target}: mappings={summary['action_mapping_count']} "
                  f"novel_candidates={transfer['statistics']['novel_candidate_count']} "
                  f"GCAD_covered={transfer['by_relation_type']['predictive_lag']['source_edges_with_mapping']}", flush=True)
    report = {
        "schema_version": 1, "generation_calls": 0, "target_data_reads": 0,
        "embedding_file_sha256": embedding_hashes,
        "config": {"min_cosine": args.min_cosine, "mapping_top_k": args.mapping_top_k,
                   "max_changed_endpoints": args.max_changed_endpoints},
        "mapping_statistics": {d: m["statistics"] for d, m in dataset_mappings.items()},
        "mapping_threshold_sensitivity": {
            dataset: {str(threshold): sum(m["cosine"] >= threshold for m in mappings["mappings"])
                      for threshold in (0.65, 0.8, 0.9)}
            for dataset, mappings in dataset_mappings.items()
        },
        "tasks": summaries,
        "interpretation": "source-only candidate feasibility; counts are not validated target relations or performance gains",
        "environment_limitation": "all mappings currently use catalogue compatibility; none has independent target-environment preference evidence",
        "implementation_sha256": {
            path.resolve().relative_to(REPOSITORY_ROOT).as_posix(): file_sha256(path)
            for path in (Path(__file__).resolve(), REPOSITORY_ROOT / "SmartGen/relation_transfer.py",
                         REPOSITORY_ROOT / "relation_research/build_embeddings.py")
        },
    }
    atomic_write_json(output / "feasibility_summary.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
