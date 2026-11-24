"""Native Windows bridge for WSL experiments when Win32 interop is disabled.

No alternate model/prompt/parser or synthetic response fallback. It only calls
the existing authenticated CodexClient and retains stdout/stderr/errors.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SmartGen"))
from codex_backend import CodexClient
from experiment import atomic_write_json, atomic_write_text


def generation_policy(root):
    """An ancestor policy applies even when the worker targets a child run."""
    root = Path(root).resolve()
    for directory in (root, *root.parents):
        policy_path = directory / "generation-policy.json"
        if policy_path.exists():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            return (directory / policy.get("queue_root", ".")).resolve(), policy
    return None, None


def request_allowed(path, policy_root, policy):
    if policy is None:
        return True
    if policy.get("hold", False):
        return False
    try:
        relative = Path(path).resolve().relative_to(policy_root)
    except ValueError:
        return False
    parts = relative.parts
    return (len(parts) == 6 and parts[3] == "codex" and parts[-1] == "request.json"
            and ('allowed_call_ids' not in policy or parts[4] in policy['allowed_call_ids'])
            and "/".join(parts[:3]) in policy.get("allowed_runs", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codex-bin", required=True)
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--idle-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    policy_root, policy = generation_policy(args.root)
    if policy is not None:
        if policy.get("hold", False):
            print(f"NO GENERATION: policy hold: {policy.get('hold_reason', 'human approval required')}", flush=True)
            return
        args.workers = min(args.workers, policy.get("max_workers", 1))
        args.max_calls = min(args.max_calls, policy["max_additional_calls"])
    lock = threading.Lock()
    stopped = threading.Event()
    state = {"count": 0, "last": time.time()}

    def work(worker):
        while True:
            directory = request = None
            with lock:
                current_root, current_policy = generation_policy(args.root)
                if (stopped.is_set() or (args.root/"generation-halt.json").exists()
                        or (current_policy is not None and current_policy.get("hold", False))):
                    return
                if state["count"] >= args.max_calls or time.time()-state["last"] >= args.idle_seconds:
                    return
                pending = [p for p in args.root.glob("**/request.json")
                    if request_allowed(p, current_root, current_policy)
                    and not any((p.parent/name).exists() for name in ("response.txt","worker-error.json","worker-claim.json"))]
                pending.sort(key=lambda p: read_priority(p))
                for request_path in pending:
                    try:
                        with (request_path.parent/"worker-claim.json").open("x",encoding="utf-8") as claim:
                            json.dump({"pid":os.getpid(),"worker":worker,"started_unix":time.time()},claim)
                    except FileExistsError:
                        continue
                    directory = request_path.parent
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    state["count"] += 1
                    state["last"] = time.time()
                    break
            if directory is None:
                time.sleep(1)
                continue
            prompt_path = directory / "prompt.txt"
            started = time.time()
            try:
                if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != request["prompt_sha256"]:
                    raise ValueError("queued prompt hash changed")
                client = CodexClient(model=request["model"], executable=args.codex_bin,
                    reasoning_effort=request["reasoning_effort"], working_directory=directory.parents[1].resolve(),
                    audit_directory=directory)
                print(f"Native REAL Codex worker={worker}: {directory}", flush=True)
                response = client.generate(prompt_path.read_text(encoding="utf-8"))
                atomic_write_text(directory / "response.txt", response)
                atomic_write_json(directory/"worker-timing.json",{"seconds":time.time()-started})
                print(f"Saved {len(client.parse_sequence_block(response))} sequences: {directory}", flush=True)
            except Exception as exc:
                atomic_write_json(directory / "worker-error.json", {"type": type(exc).__name__, "error": str(exc)})
                print(f"Codex error preserved: {exc}", flush=True)
                # Never drain the queue after a quota/transport/parser failure.
                # Existing logs/claims are retained; no automatic retry or model switch.
                stopped.set()
                atomic_write_json(args.root/"generation-halt.json", {
                    "failed_call":str(directory),"type":type(exc).__name__,"error":str(exc),
                    "resume_requires_explicit_review":True})
            with lock:
                state["last"] = time.time()

    def read_priority(path):
        return (json.loads(path.read_text(encoding="utf-8")).get("priority", []),str(path))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work,range(args.workers)))
    print(f"Native worker finished: {state['count']} attempted calls", flush=True)


if __name__ == "__main__":
    main()
