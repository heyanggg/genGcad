"""Small prompt view of a full, independently archived relation artifact."""
from __future__ import annotations

import json


def render_relation_guidance(artifact: dict) -> str:
    if artifact.get("mode") not in {"typed", "adapted"}:
        raise ValueError("unsupported relation guidance mode")
    selected = artifact["relationships"]
    if not selected:
        return ""
    actions = sorted({r[key] for r in selected for key in ("source_action", "target_action")})
    semantic_records = artifact["action_semantics"]["records"]
    payload = {
        "mode": artifact["mode"], "source_environment": artifact["source_environment"],
        "target_environment": artifact["target_environment"],
        "relationships": selected,
        "action_labels": {action: {key: semantic_records[action][key] for key in (
            "function", "capability", "desired_state", "semantic_status",
            "potential_environment_dependence",
        )} for action in actions},
    }
    return (
        "Additional typed behavior relationship audit: "
        "The original GSS and GCAD guidance remains present and unchanged. "
        "The subset below adds source-domain evidence and tentative environment policies; "
        "its selection or deferral does not remove any original guidance. "
        "GSS describes adjacent raw events. GCAD-derived lags count subsequent valid device "
        "behavior positions after non-device events are excluded; they are not hours, "
        "causal guarantees, or proof of target-environment validity. Evidence types and "
        "their lag units remain separate even when grouped under the same action pair. "
        "Action labels come from frozen, unreviewed device/command rules, not measured "
        "preferences or verified device documentation. Related functions do not imply "
        "a transition or interchangeable actions. No semantic substitutions are authorized. "
        "Hour and weekday counts describe the source only, at three-hour resolution; "
        "do not infer precise elapsed time or copy source hours as target requirements. "
        "Consider each relationship's tentative policy and the target-environment task. "
        "No source relationship is a target requirement. Do not force every "
        "relationship into every sequence. "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
