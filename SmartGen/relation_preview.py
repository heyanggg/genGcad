"""Read-only source-project preview. Outputs only to an explicit directory."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

try:
    from . import dictionary
    from .experiment import atomic_write_json, atomic_write_text
    from .relation_adapter import ENVIRONMENT_PAIRS
    from .relation_guidance import render_relation_guidance
    from .relation_stage import file_sha256, prepare_relation_stage
    from .text_translation_matrix import build_transition_guidance
except ImportError:
    import dictionary
    from experiment import atomic_write_json, atomic_write_text
    from relation_adapter import ENVIRONMENT_PAIRS
    from relation_guidance import render_relation_guidance
    from relation_stage import file_sha256, prepare_relation_stage
    from text_translation_matrix import build_transition_guidance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=["fr", "sp", "us"], required=True)
    parser.add_argument("--ori-env", choices=sorted(ENVIRONMENT_PAIRS), required=True)
    parser.add_argument("--new-env", choices=sorted(ENVIRONMENT_PAIRS.values()), required=True)
    parser.add_argument("--relation-mode", choices=["typed", "adapted"], default="adapted")
    parser.add_argument("--relation-min-support", type=int, default=3)
    parser.add_argument("--relation-max-relationships", type=int, default=12)
    parser.add_argument("--relation-max-per-target", type=int, default=4)
    parser.add_argument("--gcad-artifact", type=Path)
    parser.add_argument("--gcad-mode", choices=["auto", "off", "require"], default="auto")
    args = parser.parse_args(argv)
    if ENVIRONMENT_PAIRS[args.ori_env] != args.new_env:
        parser.error("unsupported environment transition")
    project = args.source_project.resolve()
    source_root = project / "SmartGen"
    output = args.output.resolve()
    # Never write into an upstream checkout (including source datasets).
    if output == project or project in output.parents:
        parser.error("preview output must be outside the source project")
    source = source_root / "IoT_data" / args.dataset / args.ori_env / "split_trn.pkl"
    before = file_sha256(source)
    with source.open("rb") as handle:
        sequences = pickle.load(handle)
    actions = getattr(dictionary, f"{args.dataset}_actions")
    devices = getattr(dictionary, f"{args.dataset}_devices_dict")
    # The same pure builder as ATM, with its exact ten-event cap and top-five rule.
    gss = build_transition_guidance([row[3:40:4] for row in sequences], actions)
    gcad_path = args.gcad_artifact or source_root / "artifacts/gcad" / args.dataset / args.ori_env / "gcad_hints.json"
    gcad = {"status": "missing", "lagged_behavior_relations": [],
            "disabled_reason": "no_existing_source_gcad_artifact"}
    if args.gcad_mode == "off":
        gcad.update(status="skipped", disabled_reason="disabled_by_preview_configuration")
    elif gcad_path.exists():
        gcad = json.loads(gcad_path.read_text(encoding="utf-8"))
    if args.gcad_mode == "require" and (gcad.get("status") != "ready" or not gcad.get("lagged_behavior_relations")):
        parser.error("GCAD was required but no ready artifact with relationships is available")
    artifact = prepare_relation_stage(
        source, args.dataset, args.ori_env, args.new_env, actions, devices,
        gss, gcad, output / "relation_artifact.json", mode=args.relation_mode,
        min_support=args.relation_min_support,
        max_relationships=args.relation_max_relationships,
        max_per_target=args.relation_max_per_target,
    )
    atomic_write_text(output / "guidance.txt", render_relation_guidance(artifact) + "\n")
    if file_sha256(source) != before:
        raise RuntimeError("source training changed during preview")
    summary = {"dataset": args.dataset, "source_environment": args.ori_env,
               "target_environment": args.new_env, "mode": args.relation_mode,
               "status": artifact["status"], "selection": artifact["selection"],
               "source_audit": artifact["source_audit"],
               "artifact_fingerprint": artifact["artifact_fingerprint"],
               "guidance_characters": len(render_relation_guidance(artifact)),
               "source_sha256_unchanged": before, "generation_calls": 0}
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
