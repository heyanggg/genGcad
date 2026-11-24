"""Real augmentation-only SmartGen-GCAD experiments, isolated and resumable.

Execute in the WSL smartguard_env runtime. The original checkout is read-only.
No RaSR regenerator, no endpoint transport, no environment-aware relation fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SmartGen"))

import numpy as np
import torch

from baseline1 import Anomaly_detection
from baseline2 import Train
from codex_backend import CodexClient
from dictionary import dayofweek_dict, hour_dict
from experiment import atomic_pickle_dump, atomic_write_json, atomic_write_text
from extract import Extract
from main import (ACTION_DICTIONARIES, DEVICE_DICTIONARIES, VOCABULARY_SIZE,
                  build_prompt, environment_sentence, limit_prompt_gcad_relationships)
from rasr import RaSRConfig, augment, build_relations, prepare_source, sha, split_parent_groups
from security_check import security_check
from sppc import TransformerAutoencoder
from ssc_input import compose_ssc_input, select_prompt_examples
from text_translation_matrix import build_transition_guidance
from transnumber import Transnum

ARMS = {"B0": "real-source-only", "B1": "matched-random-insert-replace",
        "B2": "semantic-only", "B3": "semantic-and-statistical",
        "B4": "disabled: independent regenerator removed by user",
        "B5": "disabled: independent regenerator removed by user"}


def load(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(args):
    return {"version": "rasr-augmentation-only-v1", "dataset": args.dataset,
            "arms": ARMS, "config": asdict(RaSRConfig()),
            "independent_regenerator": False, "target": "spring",
            "source": "winter", "calls_per_arm": args.calls,
            "examples_per_call": args.examples, "seed": 2024,
            "ssc": {"fit": "real-source-only; shared frozen checkpoint for all arms",
                    "epochs": 15, "d_model": 256, "layers": 2, "heads": 4,
                    "eval_mode": True, "selection": "same greedy SPPC retained prefix, stratified I/J",
                    "J_quota_per_prompt": args.examples // 2},
            "codex_model": args.model, "prompt_profile": "environment-aware",
            "tof_epochs": 10, "detector_epochs": 15, "detector_percentile": 95,
            "threshold": 0.915 if args.dataset == "sp" else 0.905,
            "adaptations": ["Partial borrowing of RaSR; not a full reproduction.",
                "Frozen pinned MiniLM 384D raw device/action text, not paper LLM or trainable 64D projection.",
                "No >5-item filter: original smart-home TSS predominantly has 1-2 events.",
                "Window=2 interpreted as directed adjacent pair, exactly Eq.3.",
                "Positive cosine sampling; self/invalid/None actions excluded; no semantic equivalence filter.",
                "One insertion + one semantic replacement + two statistical replacement attempts per I.",
                "Insertion inherits anchor coarse day/hour; replacement retains the same slot.",
                "At most ten events to fit existing SSC; skip duplicate/no-op/no-relation attempts.",
                "No generator, so its 64D/Adam/100 epochs/prefix K and B4/B5 no longer apply.",
                "Parent-group 90/10 source split before graph; held-out groups not used for C/T or augmentation.",
                "GSS/GCAD and shared SSC fit retain full real source data; no target data fits.",
                "Prompt-budget I/J stratification and eval-mode SSC are explicit experimental controls.",
                "Codex CLI does not expose temperature/top_p/seed; sampling is not claimed deterministic.",
                "Initial complete-pipeline pilot, not a full-scale multi-seed performance claim."]}


def ensure_edit_traces(args, output, saved):
    """Backfill non-selected edit traces for already prepared preflight runs."""
    if (output / "B3_J_unmatched.pkl").exists():
        return saved
    records = load(output / "I.pkl")
    train_ids = set(json.loads((output / "split.json").read_text())["train_ids"])
    train = [r for r in records if r["id"] in train_ids]
    actions, devices = ACTION_DICTIONARIES[args.dataset], DEVICE_DICTIONARIES[args.dataset]
    names = sorted(actions, key=actions.get)
    action_devices = [devices.get(name.split(":", 1)[0], -1) for name in names]
    candidates = [i for i, name in enumerate(names) if name.split(":", 1)[0] != "None" and action_devices[i] >= 0]
    relations = dict(np.load(output / "relations.npz"))
    raw_counts, matched_counts = {}, {}
    for arm in ("B1", "B2", "B3"):
        pool, _ = augment(train, relations, candidates, [actions[n] for n in names], action_devices,
                          RaSRConfig(), semantic_only=arm == "B2", random_control=arm == "B1")
        assert len(pool) == saved["raw_augmentation_counts"][arm]
        atomic_pickle_dump(output / f"{arm}_J_unmatched.pkl", pool)
        raw_counts[arm] = dict(Counter(r["edit"]["operation"] for r in pool))
        matched_counts[arm] = dict(Counter(r["edit"]["operation"] for r in load(output / f"{arm}_J.pkl")))
    saved.update(raw_operation_counts=raw_counts, matched_operation_counts=matched_counts)
    atomic_write_json(output / "manifest.json", saved)
    return saved


def prepare(args, out):
    output = out / "prepared"
    ready = output / "manifest.json"
    if ready.exists():
        saved = json.loads(ready.read_text())
        expected_source = args.source_root / "SmartGen/IoT_data" / args.dataset / "winter/trn.pkl"
        if (Path(saved["source_path"]).resolve() != expected_source.resolve()
                or fingerprint(expected_source) != saved["source_sha256"]
                or fingerprint(expected_source.parent / "split_trn.pkl") != saved["true_tss_sha256"]):
            raise ValueError("real source path/content changed; do not resume this experiment")
        if saved["protocol"] != protocol(args):
            old, new = dict(saved["protocol"]), dict(protocol(args))
            for key in ("calls_per_arm", "examples_per_call"):
                old.pop(key)
                new.pop(key)
            old["ssc"] = {k: v for k, v in old["ssc"].items() if k != "J_quota_per_prompt"}
            new["ssc"] = {k: v for k, v in new["ssc"].items() if k != "J_quota_per_prompt"}
            if old != new or list(out.glob("B*/codex/call-*/prompt.txt")):
                raise ValueError("protocol changed after exposure or beyond preflight budget: choose a fresh run directory")
            atomic_write_json(output / f"preflight-protocol-{sha(saved['protocol'])[:12]}.json", saved["protocol"])
            saved["protocol"] = protocol(args)
            atomic_write_json(ready, saved)
        return ensure_edit_traces(args, output, saved)
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_root / "SmartGen" / "IoT_data" / args.dataset / "winter" / "trn.pkl"
    actions, devices = ACTION_DICTIONARIES[args.dataset], DEVICE_DICTIONARIES[args.dataset]
    records, names, action_devices, candidates = prepare_source(load(source), args.dataset, actions, devices)
    config = RaSRConfig()
    train, validation = split_parent_groups(records, config)
    train_groups, val_groups = {r["parent_group"] for r in train}, {r["parent_group"] for r in validation}
    assert not train_groups & val_groups
    embeddings = json.loads((ROOT / "relation_research/embeddings/rasr-raw-v1" / f"{args.dataset}_embeddings.json").read_text())
    if embeddings["model"]["text_profile"] != "raw-action":
        raise ValueError("only unannotated raw-action embeddings are allowed")
    vectors = dict(zip(embeddings["action_names"], embeddings["vectors"]))
    relations = build_relations(train, [vectors[name] for name in names], config)
    np.savez_compressed(output / "relations.npz", **relations)
    action_ids = [actions[name] for name in names]
    pools, counts, skips = {}, {}, {}
    for arm in ("B1", "B2", "B3"):
        pools[arm], skips[arm] = augment(train, relations, candidates, action_ids, action_devices,
                                        config, semantic_only=arm == "B2", random_control=arm == "B1")
        counts[arm] = len(pools[arm])
        atomic_pickle_dump(output / f"{arm}_J_unmatched.pkl", pools[arm])
    # Exact per-parent count match across every augmentation arm, no invented variants.
    grouped = {arm: {} for arm in pools}
    for arm, pool in pools.items():
        for record in pool:
            grouped[arm].setdefault(record["parent_id"], []).append(record)
    matched = {arm: [] for arm in pools}
    quotas = {}
    for parent in sorted({r["parent_id"] for pool in pools.values() for r in pool}):
        quota = min(len(grouped[arm].get(parent, [])) for arm in pools)
        if quota:
            quotas[parent] = quota
            for arm in pools:
                # Balanced deterministic selection, not always the first semantic variants.
                options = list(grouped[arm][parent])
                random.Random(f"2024:{parent}:{arm}").shuffle(options)
                matched[arm].extend(options[:quota])
    atomic_pickle_dump(output / "I.pkl", records)
    atomic_pickle_dump(output / "source_for_ssc.pkl", [r["row"] for r in records])
    for arm, pool in matched.items():
        atomic_pickle_dump(output / f"{arm}_J.pkl", pool)
    atomic_write_json(output / "split.json", {"train_parent_groups": sorted(train_groups),
        "validation_parent_groups": sorted(val_groups), "train_ids": [r["id"] for r in train],
        "validation_ids": [r["id"] for r in validation], "matched_parent_quotas": quotas})
    real_tss = [r["row"] for r in records]
    original_tss = args.source_root / "SmartGen/IoT_data" / args.dataset / "winter/split_trn.pkl"
    assert real_tss == load(original_tss), "TSS refactor differs from existing true source data"
    gss = build_transition_guidance([r[3::4][:10] for r in real_tss], actions)
    atomic_write_json(output / "gss.json", gss)
    caches = sorted((args.source_root / "experiment_archive/formal").glob(
        f"{args.dataset}_winter_to_spring_*/artifacts/gcad/gcad_hints.json"))
    if len(caches) != 1:
        raise ValueError(f"expected one unambiguous source GCAD cache, found {caches}")
    gcad = json.loads(caches[0].read_text())
    # Verify archived GCAD was actually fit on this unchanged full real TSS.
    if fingerprint(original_tss) not in json.dumps(gcad):
        raise ValueError("GCAD cache lacks matching real-source SHA256; refit/inspect before using")
    atomic_write_json(output / "gcad-full.json", gcad)
    atomic_write_json(output / "gcad-prompt.json", limit_prompt_gcad_relationships(gcad))
    manifest = {"protocol": protocol(args), "source_path": str(source), "source_sha256": fingerprint(source),
        "true_tss_sha256": fingerprint(original_tss), "gcad_cache": str(caches[0]),
        "gcad_cache_sha256": fingerprint(caches[0]), "source_count": len(records),
        "valid_count": sum(r["valid"] for r in records), "train_count": len(train),
        "validation_count": len(validation), "raw_augmentation_counts": counts,
        "matched_augmentation_counts": {arm: len(pool) for arm, pool in matched.items()},
        "raw_operation_counts": {arm: dict(Counter(r["edit"]["operation"] for r in pool)) for arm, pool in pools.items()},
        "matched_operation_counts": {arm: dict(Counter(r["edit"]["operation"] for r in pool)) for arm, pool in matched.items()},
        "skipped": skips, "embedding_fingerprint": embeddings["artifact_fingerprint"],
        "matrix_sha256": fingerprint(output / "relations.npz"),
        "matrix_nonzero": {key: int(np.count_nonzero(relations[key])) for key in ("C", "T", "S")}}
    atomic_write_json(ready, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def ssc_model(args, out):
    checkpoint = out / "prepared/ssc.pth"
    receipt = checkpoint.with_suffix(".json")
    if not receipt.exists():
        Train(args.dataset, "winter", VOCABULARY_SIZE[args.dataset],
              input_path=out / "prepared/source_for_ssc.pkl", checkpoint_path=checkpoint)
        atomic_write_json(receipt, {"fit_input_sha256": fingerprint(out / "prepared/source_for_ssc.pkl"),
            "checkpoint_sha256": fingerprint(checkpoint), "epochs": 15, "source_only": True})
    recorded = json.loads(receipt.read_text())
    if (recorded["checkpoint_sha256"] != fingerprint(checkpoint)
            or recorded["fit_input_sha256"] != fingerprint(out / "prepared/source_for_ssc.pkl")):
        raise ValueError("shared source-only SSC checkpoint or its real input changed")
    model = TransformerAutoencoder(VOCABULARY_SIZE[args.dataset], d_model=256, nhead=4,
                                   num_encoder_layers=2, num_decoder_layers=2)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def run_arm(args, out, arm, model):
    if arm in {"B4", "B5"}:
        raise ValueError(ARMS[arm])
    run = out / arm
    result_path = run / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    run.mkdir(parents=True, exist_ok=True)
    runtime = run / "runtime"
    runtime.mkdir(exist_ok=True)
    prepared = out / "prepared"
    originals = load(prepared / "I.pkl")
    augmented = load(prepared / f"{arm}_J.pkl") if arm != "B0" else []
    pool = compose_ssc_input(originals, augmented, mode="I+J" if augmented else "I")
    atomic_write_json(run / "ssc_input.json", {"mode": "I+J" if augmented else "I",
        "counts": dict(Counter(r["origin"] for r in pool)), "record_ids_sha256": sha([r["id"] for r in pool])})
    gss, gcad = [json.loads((prepared / f"{name}.json").read_text()) for name in ("gss", "gcad-prompt")]
    actions, devices = ACTION_DICTIONARIES[args.dataset], DEVICE_DICTIONARIES[args.dataset]
    dictionaries = [dayofweek_dict, hour_dict, devices, actions]
    device_control_dict = (ROOT / "SmartGen" / f"{args.dataset}_keys_best.txt").read_text(encoding="utf-8")
    inverses = [{value: key for key, value in d.items()} for d in dictionaries]
    threshold = 0.915 if args.dataset == "sp" else 0.905
    artifact_model = f"rasr-{arm}"
    target_dir = runtime / f"IoT_data/{args.dataset}/spring"
    target_dir.mkdir(parents=True, exist_ok=True)
    (runtime / f"filter_data/{args.dataset}/spring").mkdir(parents=True, exist_ok=True)
    # Days fixed across arms; neither target data nor detector results choose examples.
    audits = []
    for call in range(args.calls):
        selected, audit = select_prompt_examples(pool, model, VOCABULARY_SIZE[args.dataset], threshold,
            day=call % 7, examples=args.examples, synthetic=arm != "B0")
        textual = [[inverses[i % 4][value] for i, value in enumerate(r["row"])] for r in selected]
        prompt = build_prompt(device_control_dict, environment_sentence("winter", "spring"), textual,
                              gss, gcad, target_environment="spring", prompt_profile="environment-aware")
        call_dir = run / f"codex/call-{call:02d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = call_dir / "prompt.txt"
        if prompt_path.exists() and prompt_path.read_text() != prompt:
            raise ValueError("resumed prompt changed")
        atomic_write_text(prompt_path, prompt)
        audit.update({"call": call, "day": call % 7, "prompt_sha256": fingerprint(prompt_path),
                      "gss_sha256": fingerprint(prepared / "gss.json"),
                      "gcad_sha256": fingerprint(prepared / "gcad-prompt.json"),
                      "actual_text_examples": textual})
        atomic_write_json(call_dir / "input_audit.json", audit)
        response_path = call_dir / "response.txt"
        if not response_path.exists():
            print(f"REAL CODEX {args.dataset} {arm} call={call+1}/{args.calls}", flush=True)
            started = time.time()
            if os.name == "posix" and args.codex_bin.endswith(".exe"):
                # WSL interoperability is disabled on this host. A native Windows
                # worker invokes exactly the existing CLI with this saved prompt.
                atomic_write_json(call_dir / "request.json", {"model": args.model,
                    "prompt_sha256": fingerprint(prompt_path), "reasoning_effort": "none"})
                while not response_path.exists():
                    error = call_dir / "worker-error.json"
                    if error.exists():
                        raise RuntimeError(error.read_text())
                    if time.time() - started > 1000:
                        raise TimeoutError("native Codex worker response timeout; inspect request and worker log")
                    time.sleep(1)
                response = response_path.read_text()
            else:
                client = CodexClient(model=args.model, executable=args.codex_bin, timeout=900,
                                     working_directory=run, audit_directory=call_dir)
                response = client.generate(prompt)
                atomic_write_text(response_path, response)
            atomic_write_json(call_dir / "timing.json", {"seconds": time.time() - started,
                                   "generated_count": len(CodexClient.parse_sequence_block(response))})
        response = response_path.read_text()
        CodexClient.parse_sequence_block(response)
        atomic_pickle_dump(target_dir / f"{args.dataset}_spring_generation_day_{call}_SPPC_th={threshold}_{artifact_model}.pkl", response)
        audits.append(audit)
    atomic_write_json(run / "prompt_exposure.json", {"codex_calls": len(audits),
        "actual_total_examples": sum(sum(a["actual_prompt_origin_counts"].values()) for a in audits),
        "actual_J_examples": sum(a["actual_prompt_origin_counts"].get("J", 0) for a in audits),
        "actual_unique_ids": sorted({i for a in audits for i in a["actual_prompt_ids"]}), "calls": audits})
    # Target normal/attack data are first accessed here, AFTER all fitting and generation.
    for relative in [f"IoT_data/{args.dataset}/spring/split_test.pkl",
                     f"attack/{args.dataset}/labeled_{args.dataset}_spring_attack_heater.pkl"]:
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source_root / "SmartGen" / relative, destination)
    previous = Path.cwd()
    os.chdir(runtime)
    try:
        Extract(args.dataset, "spring", threshold, "SPPC", artifact_model, list(range(args.calls)))
        Transnum(args.dataset, "spring", threshold, "SPPC", artifact_model, list(range(args.calls)), dictionaries)
        tof = security_check(args.dataset, "spring", threshold, "SPPC", artifact_model, keep_intermediates=True)
        detection = Anomaly_detection(args.dataset, "spring", threshold, "SPPC", artifact_model, 95)
    finally:
        os.chdir(previous)
    result = {"dataset": args.dataset, "arm": arm, "status": "completed", "tof": tof,
              "detection": detection, "actual_J_prompt_examples": sum(a["actual_prompt_origin_counts"].get("J", 0) for a in audits)}
    atomic_write_json(result_path, result)
    print(f"COMPLETED {args.dataset} {arm}: {json.dumps(detection['metrics'])}", flush=True)
    return result


def validate_ssc_inputs(args, out, model):
    """Exercise B0-B3 SSC interfaces with no extra Codex calls."""
    originals = load(out / "prepared/I.pkl")
    audits = {}
    threshold = 0.915 if args.dataset == "sp" else 0.905
    for arm in ("B0", "B1", "B2", "B3"):
        j = [] if arm == "B0" else load(out / f"prepared/{arm}_J.pkl")
        pool = compose_ssc_input(originals, j, mode="I" if arm == "B0" else "I+J")
        audits[arm] = []
        for day in range(args.calls):
            _, audit = select_prompt_examples(pool, model, VOCABULARY_SIZE[args.dataset], threshold,
                day=day, examples=args.examples, synthetic=arm != "B0")
            audits[arm].append(audit)
    atomic_write_json(out / "ssc-interface-preflight.json", {"codex_calls": 0, "arms": audits})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sp", "us"], required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/home/heyang/projects/SmartGen_GCAD"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=["B0", "B3"])
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex-bin", default="/mnt/c/Users/heyang/AppData/Local/OpenAI/Codex/bin/23f7d7f110b19ac3/codex.exe")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args(argv)
    if args.calls < 1 or args.calls > 7:
        parser.error("calls must be in 1..7 (one fixed day per call)")
    if any(arm in {"B4", "B5"} for arm in args.arms):
        parser.error("B4/B5 were cancelled by user: independent regenerator forbidden")
    torch.set_num_threads(2)
    out = args.output.resolve() / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        prepare(args, out)
        if args.prepare_only:
            return 0
        model = ssc_model(args, out)
        if args.validate_inputs_only:
            validate_ssc_inputs(args, out, model)
            print("B0-B3 SSC input validation passed; zero additional Codex calls", flush=True)
            return 0
        results = [run_arm(args, out, arm, model) for arm in args.arms]
        summary = {"results": results, "seconds": time.time()-started}
        if not (out / "summary.json").exists():
            atomic_write_json(out / "summary.json", summary)
        else:
            atomic_write_json(out / "resume_check.json", {"status": "validated_cached_results",
                "seconds": time.time()-started, "additional_codex_calls": 0, "arms": args.arms})
        atomic_write_json(out / "execution_status.json", {"status": "completed", "arms": args.arms,
            "historical_failure_record": "failure.json" if (out / "failure.json").exists() else None})
    except Exception as exc:
        failure = {"type": type(exc).__name__, "error": str(exc), "seconds": time.time()-started}
        atomic_write_json(out / "failure.json", failure)
        atomic_write_json(out / f"failures/{time.time_ns()}.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
