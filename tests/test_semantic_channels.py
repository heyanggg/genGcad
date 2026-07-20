import numpy as np
import pytest

from SmartGen.gcad_source.event_tensorizer import SourceEventTensorizer
from SmartGen.gcad_source.gss_fusion import fuse_gss
from SmartGen.gcad_source.prompt_adapter import adapt_prompt
from SmartGen.gcad_source.stability_filter import build_stable_relation


def invalid_relation():
    return {"edges": [{"source": "Light:switch on", "target": "None:location", "stable_score": 1.0}]}


def test_none_location_cannot_enter_channel_vocabulary():
    tensorizer = SourceEventTensorizer(
        device_names={1: "Light", 2: "None"},
        action_names={10: "Light:switch on", 20: "None:location"},
        history_length=1,
    )
    result = tensorizer.tensorize([[0, 0, 2, 20, 0, 1, 1, 10]])
    assert result.vocabulary == ["Light:switch on"]
    assert any(item["reason"] == "invalid_semantic_event" for item in result.skipped_sequences)


def test_invalid_channel_rejected_by_stability_fusion_and_prompt():
    with pytest.raises(ValueError, match="None:location"):
        build_stable_relation([np.eye(2)], ["Light:switch on", "None:location"])
    with pytest.raises(ValueError, match="None:location"):
        fuse_gss({}, invalid_relation(), {"Light": ["switch on"]})
    with pytest.raises(ValueError, match="None:location"):
        adapt_prompt("baseline", True, invalid_relation(), {})
