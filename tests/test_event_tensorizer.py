import numpy as np

from SmartGen.gcad_source.event_tensorizer import SourceEventTensorizer
from SmartGen.gcad_source.window_dataset import SequenceWindowDataset


def tensorizer(**kwargs):
    return SourceEventTensorizer(
        history_length=kwargs.pop("history_length", 1),
        device_names={1: "Light", 2: "Blind"},
        action_names={10: "Light:switch on", 11: "Light:switch off", 20: "Blind:windowShade open"},
        **kwargs,
    )


def test_vocabulary_is_deterministic():
    a = [0, 1, 2, 20, 0, 2, 1, 10]
    b = [0, 1, 1, 11]
    assert tensorizer().tensorize([a, b]).vocabulary == tensorizer().tensorize([b, a]).vocabulary


def test_windows_never_cross_sequence_boundaries():
    result = tensorizer().tensorize([[0, 0, 1, 10, 0, 1, 1, 11], [0, 0, 2, 20, 0, 1, 2, 20]])
    dataset = SequenceWindowDataset(result.sequences, history_length=1)
    assert len(dataset) == 2
    assert [dataset[i][3] for i in range(len(dataset))] == [0, 1]
    assert dataset[0][0][0].argmax().item() != dataset[1][0][0].argmax().item()


def test_multiple_events_same_slot_binary_and_count():
    sequence = [0, 0, 1, 10, 0, 0, 1, 10]
    binary = tensorizer(occurrence_mode="binary").tensorize([sequence]).sequences[0]
    count = tensorizer(occurrence_mode="count").tensorize([sequence]).sequences[0]
    assert binary.sum() == 1
    assert count.sum() == 2


def test_masks_and_short_sequence_are_explicit():
    result = tensorizer(history_length=2).tensorize([[0, 0, 1, 10]])
    assert result.sequences[0].shape == (1, 1)
    assert not result.valid_window_masks[0].any()
    assert result.metadata["valid_window_count"] == 0


def test_week_wrap_is_unwrapped_inside_one_sequence():
    result = tensorizer().tensorize([[6, 7, 1, 10, 0, 0, 1, 11]])
    assert result.sequences[0].shape[0] == 2

