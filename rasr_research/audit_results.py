"""Verify real prompt exposure, split/count controls and retain all pilot results."""
import argparse
import hashlib
import json
import pickle
import sys
from importlib.metadata import version
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SmartGen"))
from codex_backend import CodexClient
import dictionary
from experiment import atomic_write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def audit(root, require_complete=False):
    report = {"protocol": "augmentation-only", "independent_regenerator": False,
              "all_results_retained": True, "datasets": {},
              "limitation": "Two calls per arm, one seed, very small generated training sets. No significance or full-scale efficacy claim."}
    report["runtime"] = {"python": sys.version, "torch": version("torch"), "numpy": version("numpy")}
    report["audited_current_implementation_sha256"] = {
        relative: sha(ROOT / relative) for relative in ("SmartGen/rasr.py", "SmartGen/ssc_input.py",
            "SmartGen/split.py", "SmartGen/baseline2.py", "SmartGen/codex_backend.py",
            "SmartGen/main.py", "SmartGen/security_check.py", "SmartGen/baseline1.py",
            "rasr_research/run_experiment.py", "rasr_research/codex_worker.py")}
    for dataset in ("sp", "us"):
        directory = root / dataset
        prepared = directory / "prepared"
        manifest = read(prepared / "manifest.json")
        config = manifest["protocol"]
        split = read(prepared / "split.json")
        train_groups = set(split["train_parent_groups"])
        assert not set(split["train_parent_groups"]) & set(split["validation_parent_groups"])
        assert not config["independent_regenerator"]
        source = Path(manifest["source_path"])
        assert sha(source) == manifest["source_sha256"], "real source file changed"
        true_tss = source.parent / "split_trn.pkl"
        assert sha(true_tss) == manifest["true_tss_sha256"], "real TSS file changed"
        counts, augmentations = {}, {}
        original = {r["id"]: r for r in unpickle(prepared / "I.pkl")}
        devices = getattr(dictionary, f"{dataset}_devices_dict")
        action_names = {value: name for name, value in getattr(dictionary, f"{dataset}_actions").items()}
        for arm in ("B1", "B2", "B3"):
            rows = unpickle(prepared / f"{arm}_J.pkl")
            counts[arm] = Counter(r["parent_id"] for r in rows)
            assert all(r["parent_group"] in train_groups for r in rows)
            for r in rows:
                assert r["parent_id"] in original
                parent = original[r["parent_id"]]
                slots = {tuple(parent["row"][i:i+2]) for i in range(0, len(parent["row"]), 4)}
                assert 0 < len(r["row"]) <= 40 and len(r["row"]) % 4 == 0
                assert r["tokens"] != parent["tokens"]
                for offset in range(0, len(r["row"]), 4):
                    day, hour, device, action = r["row"][offset:offset+4]
                    assert (day, hour) in slots
                    assert devices[action_names[action].split(":", 1)[0]] == device
            assert max(counts[arm].values(), default=0) <= 4
            augmentations[arm] = {r["id"]: r for r in rows}
        assert counts["B1"] == counts["B2"] == counts["B3"]
        real_rows = {tuple(r["row"]) for r in original.values()}
        row = {"source_count": manifest["source_count"], "source_unchanged": True,
               "matched_augmentation_count_each_arm": sum(counts["B3"].values()),
               "split_disjoint": True, "all_matched_J_legal_with_original_time_slots": True, "arms": {}}
        for arm in ("B0", "B3"):
            run = directory / arm
            result_path = run / "result.json"
            if not result_path.exists():
                row["arms"][arm] = {"status": "not-yet-completed"}
                if require_complete:
                    raise ValueError(f"missing real completed result: {dataset} {arm}")
                continue
            result = read(result_path)
            exposure = read(run / "prompt_exposure.json")
            assert exposure["codex_calls"] == config["calls_per_arm"]
            assert exposure["actual_total_examples"] == config["calls_per_arm"] * config["examples_per_call"]
            operations = Counter()
            novel_j, raw_sequences, raw_events = 0, 0, 0
            for call in exposure["calls"]:
                call_dir = run / f"codex/call-{call['call']:02d}"
                prompt = (call_dir / "prompt.txt").read_text(encoding="utf-8")
                assert sha(call_dir / "prompt.txt") == call["prompt_sha256"]
                assert str(call["actual_text_examples"]) in prompt, "examples not actually embedded in prompt"
                assert call["gss_sha256"] == sha(prepared / "gss.json")
                assert call["gcad_sha256"] == sha(prepared / "gcad-prompt.json")
                receipt = read(call_dir / "exit.json")
                assert receipt["returncode"] == 0 and receipt["protocol"]["model"] == config["codex_model"]
                response = CodexClient.parse_sequence_block((call_dir / "response.txt").read_text(encoding="utf-8"))
                assert response
                raw_sequences += len(response)
                raw_events += sum(len(r)//4 for r in response)
                for identifier in call["actual_prompt_ids"]:
                    if identifier not in original:
                        assert identifier in augmentations[arm]
                        operations[augmentations[arm][identifier]["edit"]["operation"]] += 1
                        novel_j += tuple(augmentations[arm][identifier]["row"]) not in real_rows
            expected_j = 0 if arm == "B0" else config["calls_per_arm"] * config["examples_per_call"] // 2
            assert exposure["actual_J_examples"] == expected_j
            row["arms"][arm] = {"status": result["status"], "generated": result["tof"]["input_count"],
                "after_tof": result["tof"]["final_count"], "actual_J_prompt_examples": exposure["actual_J_examples"],
                "J_prompt_operations": dict(operations), "metrics": result["detection"]["metrics"],
                "detector_train_count": result["detection"]["training_sequence_count"],
                "detector_validation_count": result["detection"]["validation_sequence_count"],
                "J_examples_novel_vs_all_real_rows": novel_j,
                "raw_codex_sequences": raw_sequences, "raw_codex_events": raw_events,
                "result_path": str(result_path.resolve())}
            threshold = config["threshold"]
            numeric = unpickle(run / f"runtime/filter_data/{dataset}/spring/{dataset}_spring_generation_SPPC_th={threshold}_rasr-{arm}_seq.pkl")
            row["arms"][arm]["numeric_events_after_existing_dictionary_conversion"] = sum(len(r)//4 for r in numeric)
        if all(row["arms"][arm]["status"] == "completed" for arm in ("B0", "B3")):
            row["B3_minus_B0"] = {key: row["arms"]["B3"]["metrics"][key]-row["arms"]["B0"]["metrics"][key]
                                    for key in ("f1_score", "auroc", "auprc", "FPR")}
            assert sha(prepared / "ssc.pth") == read(prepared / "ssc.json")["checkpoint_sha256"]
        report["datasets"][dataset] = row
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, args.require_complete)
    atomic_write_json(args.root / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
