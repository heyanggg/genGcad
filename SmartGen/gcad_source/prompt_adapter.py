from __future__ import annotations

import json


SOFT_RELATION_PREAMBLE = (
    "\n\nOptional source-only predictive relation guidance: The following patterns were learned "
    "only from normal behavior in the source environment. They are soft generation hints, "
    "not observations of the target environment and not physical causal laws. The target "
    "context description and static device-action legality take priority. Do not mechanically "
    "cover every relation.\n"
)


def adapt_prompt(base_prompt: str, enabled: bool, stable_relation: dict | None = None, fused_gss: dict | None = None) -> str:
    if not enabled:
        return base_prompt
    if stable_relation is None or fused_gss is None:
        raise ValueError("enabled GCAD prompt adapter requires stable_relation and fused_gss")
    compact_edges = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "strength": round(float(edge.get("stable_score", edge.get("strength", 0))), 6),
            "lag": edge.get("mean_primary_lag", edge.get("primary_lag")),
        }
        for edge in stable_relation.get("edges", [])[:30]
    ]
    return (
        base_prompt
        + SOFT_RELATION_PREAMBLE
        + "Stable source predictive directions: "
        + json.dumps(compact_edges, ensure_ascii=False, sort_keys=True)
        + "\nConservatively fused transition guidance: "
        + json.dumps(fused_gss, ensure_ascii=False, sort_keys=True)
    )


def build_original_smartgen_prompt(
    device_control_text: str,
    context_sentence: str,
    representative_sequences,
    action_transitions: dict,
) -> str:
    """Exact baseline prompt concatenation from SmartGen/main.py lines 136-151."""
    return (
        "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. "
        "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences."
        "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices."
        f"The set of the possible device and device states: {device_control_text}"
        f"{context_sentence} The user's compressed original sequences of behavior: {representative_sequences}. User's behavior habits: {action_transitions}"
        "Your task: First, select the possible new device states from the set of devices and device states which are also possible new user behaviors. "
        "The second step is to reasonably add possible new user behaviors to the original user behavior sequences. The third step is to reasonably continue and expand the sequence based on user behavior habits."
        "Requirements:"
        "1.Please consider the devices that will be used in the new environment as widely as possible based on the set of devices."
        "2.Please strictly follow the correspondence between the devices and device states to generate. Do not generate device states that do not match the device."
        "3.Please add as many new devices and device behaviors as possible to better adapt to changes in the environment."
        "4.Please make sure that the generated sequence is not a single behavior, but a sequence of consecutive behaviors."
        "5.Please also generate reasonable behavior time when generating, not just a single behavior."
        "6.The final generated behavior sequences set is in the format of <seq [['...'], ['...'], ['...']] seq>. For example, the sequences set can be like <seq [['Sunday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(21~24)', 'Camera', 'Camera:notification', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning'], ['Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close']] seq>"
        "Note that each [...] subsequence represents the user's behavior over a period of time. There is no direct correlation between subsequences. At the same time, the final sequence is strictly generated in the format of <seq [['......'], ['......'], ['......']] seq> without line breaks or inconsistent formats."
        "Please think step by step, and return the final generated user behavior sequence set."
    )

