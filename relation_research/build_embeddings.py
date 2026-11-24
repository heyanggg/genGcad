"""Build frozen action embeddings with a pinned, local MiniLM encoder.

Only --download accesses the network. Inference always uses local_files_only
and safetensors, with no remote Python code and no target behavior data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from SmartGen import dictionary
from SmartGen.action_semantics import build_action_semantics, canonical_sha256
from SmartGen.experiment import atomic_write_json
from SmartGen.relation_stage import file_sha256


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = [
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json",
    "1_Pooling/config.json", "modules.json", "README.md",
]
TEXT_VERSION = "device-command-function-state-v1"


def action_text(action, label):
    device, _, command = action.partition(":")
    words = lambda value: re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("_", " ")
    return (f"Device: {words(device)}. Command: {words(command)}. "
            f"Function: {words(label['function'])}. Desired state: {words(label['desired_state'])}. "
            f"Original action: {action}.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--text-profile", choices=["raw-action", "annotated"], default="raw-action")
    parser.add_argument("--datasets", choices=["fr", "sp", "us"], nargs="+", default=["fr", "sp", "us"])
    args = parser.parse_args(argv)
    model_dir = args.model_dir.resolve()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES,
            local_dir=str(model_dir), local_dir_use_symlinks=False,
            cache_dir=str(model_dir.parent / "hub_cache"), token=False,
        )
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        parser.error(f"local model files missing: {missing}; use --download once")

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                                      trust_remote_code=False, use_safetensors=True).eval().cpu()
    model_files = {p.relative_to(model_dir).as_posix(): file_sha256(p)
                   for p in sorted(model_dir.rglob("*"))
                   if p.is_file() and ".cache" not in p.relative_to(model_dir).parts}
    model_metadata = {
        "model_id": MODEL_ID, "requested_revision": MODEL_REVISION,
        "model_files_sha256": model_files, "pooling": "attention-mask mean; L2 normalization",
        "text_version": "raw-device-action-v1" if args.text_profile == "raw-action" else TEXT_VERSION,
        "text_profile": args.text_profile, "max_length": 256, "device": "cpu",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "implementation_sha256": file_sha256(Path(__file__)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    for dataset in args.datasets:
        semantics = build_action_semantics(getattr(dictionary, f"{dataset}_actions"),
                                            getattr(dictionary, f"{dataset}_devices_dict"))
        names = sorted(semantics["records"])
        texts = [(f"Device: {name.partition(':')[0]}. Action: {name.partition(':')[2]}."
                  if args.text_profile == "raw-action" else action_text(name, semantics["records"][name])) for name in names]
        vectors = []
        for start in range(0, len(names), 32):
            encoded = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                                max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.tolist())
        artifact = {
            "schema_version": 1, "dataset": dataset,
            "dictionary_sha256": semantics["dictionary_sha256"],
            "semantics_sha256": canonical_sha256(semantics), "model": model_metadata,
            "action_names": names, "texts": texts, "vectors": vectors,
            "text_sha256": canonical_sha256(dict(zip(names, texts))),
            "limitations": [
                "MiniLM is a sentence encoder, not the original RaSR LLM implementation.",
                "Raw-action profile uses no functional/state annotation or environment label in text.",
                "Cosine similarity is not functional equivalence, transfer validity, or a probability.",
            ],
        }
        artifact["artifact_fingerprint"] = canonical_sha256(artifact)
        output = args.output.resolve() / f"{dataset}_embeddings.json"
        if output.exists() and json.loads(output.read_text(encoding="utf-8")) != artifact:
            raise ValueError("embedding artifact already exists with different inputs; choose a new output directory")
        if not output.exists():
            atomic_write_json(output, artifact)
        print(f"{dataset}: {len(names)} actions x {len(vectors[0])} dimensions; {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
