import numpy as np
import torch

from SmartGen.gcad_source.asymmetric_filter import asymmetric_difference
from SmartGen.gcad_source.gradient_relation import extract_gradient_relation
from SmartGen.gcad_source.mixer_predictor import SourceGCADMixer
from SmartGen.gcad_source.stability_filter import build_stable_relation


def test_each_output_channel_backpropagates_independently(tmp_path):
    model = SourceGCADMixer(2, 3, hidden_size=4, num_layers=1, dropout=0)
    x = torch.rand(4, 2, 3, requires_grad=True)
    target = torch.rand(4, 3)
    loss = model.per_channel_loss(model(x), target)
    assert loss.shape == (4, 3)
    gradients = [torch.autograd.grad(loss[:, j].sum(), x, retain_graph=True)[0] for j in range(3)]
    assert all(item.shape == x.shape for item in gradients)
    assert not torch.equal(gradients[0], gradients[1])


def test_gradient_relation_shapes_and_lags(tmp_path):
    model = SourceGCADMixer(2, 2, hidden_size=4, num_layers=1, dropout=0)
    sequence = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    result = extract_gradient_relation(model, [sequence], 2, tmp_path, ["a", "b"], batch_size=2)
    assert result["raw_relation"].shape == (2, 2)
    assert result["lag_relation"].shape == (2, 2, 2)
    assert result["metadata"]["window_count"] == 2


def test_asymmetric_difference_direction_topk_and_no_self_loops():
    raw = np.array([[9, 5, 2], [1, 8, 3], [4, 1, 7]], dtype=float)
    result = asymmetric_difference(raw, top_k=1, normalize=False)
    assert np.diag(result).sum() == 0
    assert result[0, 1] == 4
    assert result[2, 0] == 2
    assert np.count_nonzero(result, axis=1).max() <= 1


def test_stability_filter_is_deterministic_and_tracks_replicates(tmp_path):
    matrices = [np.array([[0, 0.8], [0, 0]]), np.array([[0, 0.6], [0, 0]])]
    first = build_stable_relation(
        matrices, ["a", "b"], occurrence_threshold=1, direction_consistency_threshold=1,
        stability_threshold=0.1, seeds=[1, 2], split_ids=["a", "b"], output_dir=tmp_path
    )
    second = build_stable_relation(
        matrices, ["a", "b"], occurrence_threshold=1, direction_consistency_threshold=1,
        stability_threshold=0.1, seeds=[1, 2], split_ids=["a", "b"]
    )
    assert np.array_equal(first["matrix"], second["matrix"])
    assert first["edges"][0]["seed_count"] == 2
    assert first["edges"][0]["split_count"] == 2

