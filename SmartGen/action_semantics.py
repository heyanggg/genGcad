"""Versioned, conservative action labels. No training data or LLM calls are used.

These labels are an inspectable baseline, not verified device documentation or
LLM embeddings. Unknown commands abstain. Similar function never implies a
transition, causal effect, or permission to substitute an action.
"""
from __future__ import annotations

import hashlib
import json
import re


SEMANTICS_VERSION = "device-command-rules-v1"
DEVICE_FUNCTIONS = {
    "AirConditioner": "climate_control", "Heater": "heating",
    "Fan": "air_circulation", "AirPurifier": "air_purification",
    "Light": "illumination", "Blind": "window_shading",
    "Camera": "visual_monitoring", "MotionSensor": "motion_sensing",
    "PresenceSensor": "presence_sensing", "ContactSensor": "contact_sensing",
    "SmartLock": "access_control", "GarageDoor": "garage_access",
    "WaterValve": "water_flow", "RobotCleaner": "floor_cleaning",
    "Television": "video_entertainment", "Projector": "video_entertainment",
    "NetworkAudio": "audio_entertainment", "Computer": "computing",
    "Dishwasher": "dishwashing", "Dryer": "clothes_drying",
    "Washer": "clothes_washing", "ClothingCareMachine": "clothes_care",
    "Microwave": "food_heating", "Oven": "food_cooking",
    "Refrigerator": "food_cooling", "CurbPowerMeter": "energy_metering",
    "Elevator": "vertical_transport", "Siren": "audible_alert",
    # A switch/plug's controlled appliance cannot be inferred from its name.
    "Switch": "unspecified_control", "SmartPlug": "unspecified_control",
    "RemoteController": "unspecified_control", "Other": "unknown",
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _command_state(command: str) -> tuple[str, str]:
    """Return an exact command label and its desired state (not pre-state)."""
    lower = command.lower()
    if lower.startswith("notification"):
        return "notification", "report_only"
    if lower == "refresh refresh":
        return "refresh", "request_only"
    for capability, values in (
        ("switch", {"on": "on", "off": "off", "toggle": "toggle"}),
        ("windowshade", {"open": "open", "close": "closed"}),
        ("valve", {"open": "open", "close": "closed"}),
        ("doorcontrol", {"open": "open", "close": "closed"}),
        ("lock", {"lock": "locked", "unlock": "unlocked"}),
        ("mediaplayback", {"play": "playing", "pause": "paused", "stop": "stopped"}),
        ("audiomute", {"mute": "muted", "unmute": "unmuted"}),
        ("setrobotcleanermovement", {"cleaning": "cleaning", "charging": "charging",
                                     "homing": "homing", "poweroff": "off"}),
    ):
        match = re.fullmatch(re.escape(capability) + r"\s+(\w+)", lower)
        if match and match[1] in values:
            return capability, values[match[1]]
    if re.fullmatch(r"(?:dryer|washer)operatingstate setmachinestate (run|pause|stop)", lower):
        return "machine_operation", {"run": "running", "pause": "paused", "stop": "stopped"}[lower.split()[-1]]
    if lower in {"temperatureup", "temperaturedown", "fanspeedup", "fanspeeddown"}:
        return "relative_adjustment", "increase" if lower.endswith("up") else "decrease"
    if lower.endswith("motioncommands active"):
        return "motion_event", "reported_active"
    return "unknown", "unknown"


def build_action_semantics(actions: dict, devices: dict) -> dict:
    records = {}
    for action, action_id in sorted(actions.items()):
        device, separator, command = action.partition(":")
        valid = bool(separator and device in devices and device != "None" and command)
        capability, state = _command_state(command) if valid else ("unknown", "unknown")
        function = DEVICE_FUNCTIONS.get(device, "unknown")
        resolved = valid and capability != "unknown" and function not in {"unknown", "unspecified_control"}
        records[action] = {
            "action_id": action_id, "device": device, "device_id": devices.get(device),
            "command": command,
            "valid_device_action": valid, "function": function,
            "capability": capability, "desired_state": state,
            "semantic_status": "rule_labeled" if resolved else "unresolved",
            "potential_environment_dependence": (
                ["season_and_unobserved_temperature"]
                if function in {"climate_control", "heating", "air_circulation"}
                else ["time_of_day_and_unobserved_light"]
                if function in {"illumination", "window_shading"} else []
            ),
            "provenance": {"method": "lexical_rules", "version": SEMANTICS_VERSION,
                           "review_status": "not_independently_reviewed"},
        }
    return {
        "schema_version": 1, "version": SEMANTICS_VERSION,
        "dictionary_sha256": canonical_sha256({"actions": actions, "devices": devices}),
        "records": records, "semantic_substitution_enabled": False,
        "embedding_model": None,
    }
