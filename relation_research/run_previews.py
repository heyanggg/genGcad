"""Inspect all nine source tasks without training, generation, or target reads."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from SmartGen.experiment import atomic_write_json
from SmartGen.relation_preview import main as preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gcad-mode", choices=["formal", "runtime", "off"], default="formal")
    args = parser.parse_args(argv)
    project, output = args.source_project.resolve(), args.output.resolve()
    if output == project or project in output.parents:
        parser.error("preview output must be outside the source project")
    summaries = []
    for dataset in ("fr", "sp", "us"):
        for source, target in (("winter", "spring"), ("daytime", "night"), ("single", "multiple")):
            extra = []
            if args.gcad_mode == "formal":
                candidates = sorted((project / "experiment_archive/formal").glob(
                    f"{dataset}_{source}_to_{target}_*/artifacts/gcad/gcad_hints.json"
                ))
                if len(candidates) != 1:
                    raise ValueError(f"expected one formal source GCAD artifact for {dataset}/{source}, got {len(candidates)}")
                extra = ["--gcad-artifact", str(candidates[0])]
            elif args.gcad_mode == "off":
                extra = ["--gcad-mode", "off"]
            for mode in ("typed", "adapted"):
                destination = output / f"{dataset}_{source}_to_{target}" / mode
                with contextlib.redirect_stdout(io.StringIO()):
                    preview([
                        "--source-project", str(project), "--output", str(destination),
                        "--dataset", dataset, "--ori-env", source, "--new-env", target,
                        "--relation-mode", mode, *extra,
                    ])
                summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
                summaries.append(summary)
                selection = summary["selection"]
                print(f"{dataset} {source}->{target} {mode}: selected={selection['selected_count']} "
                      f"deferred={selection['deferred_count']} gcad={summary['source_audit']['gcad_status']}")
    result = {"source_project": str(project), "gcad_mode": args.gcad_mode,
              "generation_calls": 0, "target_data_reads": 0,
              "interpretation": "source-only interface and coverage audit; not a performance experiment",
              "tasks": summaries}
    atomic_write_json(output / "matrix_summary.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
