from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

from SmartGen import dictionary

from .data_boundary import require_roles
from .data_roles import DataRole, RoleBoundPath
from .prompt_adapter import adapt_prompt, build_original_smartgen_prompt


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mappings(dataset: str):
    devices = getattr(dictionary, f"{dataset}_devices_dict")
    actions = getattr(dictionary, f"{dataset}_actions")
    return [
        dictionary.dayofweek_dict,
        dictionary.hour_dict,
        devices,
        actions,
    ]


def numeric_to_text(sequences, dataset: str):
    mappings = _mappings(dataset)
    inverse = [{value: key for key, value in mapping.items()} for mapping in mappings]
    return [
        [inverse[index % 4][int(value)] for index, value in enumerate(sequence)]
        for sequence in sequences
    ]


def build_cell_prompts(
    *,
    dataset: str,
    source_context: str,
    target_context: str,
    compression_threshold: float,
    stable_relation_path: str | Path,
    fused_gss_path: str | Path,
    output_dir: str | Path,
    source_root: str | Path = "SmartGen/IoT_data",
    device_control_path: str | Path | None = None,
) -> dict:
    source_directory = Path(source_root) / dataset / source_context
    source_files = [source_directory / f"trn_day_{day}_SPPC_th={compression_threshold}.pkl" for day in range(7)]
    artifacts = [RoleBoundPath.build(path, DataRole.SOURCE_NORMAL) for path in source_files]
    require_roles("prompt", artifacts)
    representatives = []
    counts = []
    for path in source_files:
        with path.open("rb") as handle:
            values = pickle.load(handle)
        counts.append(len(values))
        representatives.extend(numeric_to_text(values, dataset))
    gss_path = source_directory / "action_transitions.json"
    original_gss = json.loads(gss_path.read_text(encoding="utf-8"))
    stable = json.loads(Path(stable_relation_path).read_text(encoding="utf-8"))
    fused = json.loads(Path(fused_gss_path).read_text(encoding="utf-8"))
    if device_control_path is None:
        device_control_path = Path("SmartGen") / f"{dataset}_keys_best.txt"
    device_control = Path(device_control_path).read_text(encoding="utf-8")
    if target_context == "spring":
        sentence = f"The previous environment is {source_context}. The changed environment is warm {target_context}."
    elif target_context == "night":
        sentence = (
            f"The previous environment: user is active during the {source_context} and rest at {target_context}. "
            f"The changed environment: user is active at {target_context} and rest during the {source_context}."
        )
    elif target_context == "multiple":
        sentence = (
            f"The previous environment was for a {source_context} person to be at home, and the changed "
            f"environment is for {target_context} people to be at home"
        )
    else:
        raise ValueError("unsupported SmartGen target context")
    baseline = build_original_smartgen_prompt(device_control, sentence, representatives, original_gss)
    enhanced = adapt_prompt(baseline, True, stable, fused)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "baseline_prompt.txt"
    enhanced_path = output / "gcad_gss_prompt.txt"
    baseline_path.write_text(baseline, encoding="utf-8")
    enhanced_path.write_text(enhanced, encoding="utf-8")
    manifest = {
        "dataset": dataset,
        "source_context": source_context,
        "target_context": target_context,
        "compression_threshold": compression_threshold,
        "representative_sequence_count": len(representatives),
        "representative_counts_by_day": counts,
        "source_files": [{"path": str(path.resolve()), "sha256": _sha(path)} for path in source_files],
        "original_gss_path": str(gss_path.resolve()),
        "original_gss_sha256": _sha(gss_path),
        "stable_relation_path": str(Path(stable_relation_path).resolve()),
        "fused_gss_path": str(Path(fused_gss_path).resolve()),
        "baseline_prompt_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "gcad_gss_prompt_sha256": hashlib.sha256(enhanced.encode()).hexdigest(),
        "baseline_is_original_prompt_builder_without_gcad_block": True,
        "uses_target_behavior": False,
    }
    (output / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

