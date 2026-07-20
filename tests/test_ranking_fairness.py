import pickle

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from anomaly_detection_pipeline.Anomaly_Detection_pipeline_model import make_data, masked_weighted_loss
from SmartGen.gcad_source.ranking_weights import bounded_relation_weights


def rows(scores):
    return [{"sequence_index": index, "weighted_consistency": score} for index, score in enumerate(scores)]


def test_uniform_relation_scores_bypass_with_unit_diagnostics():
    weights, diagnostics = bounded_relation_weights(rows([0.0, 0.0, 0.0]))
    assert weights is None
    assert diagnostics["ranking_signal_absent"] is True
    assert diagnostics["mean"] == diagnostics["min"] == diagnostics["max"] == 1.0
    assert diagnostics["effective_sample_size"] == 3.0


def test_nonuniform_weights_are_mean_one_bounded_and_have_correct_ess_entropy():
    weights, diagnostics = bounded_relation_weights(rows([-2.0, 0.0, 1.0, 3.0]))
    values = np.asarray(list(weights.values()))
    assert np.isclose(values.mean(), 1.0)
    assert values.min() >= 0.9 and values.max() <= 1.1
    expected_ess = values.sum() ** 2 / np.sum(values ** 2)
    assert np.isclose(diagnostics["effective_sample_size"], expected_ess)
    assert 0 < diagnostics["normalized_weight_entropy"] <= 1


def test_baseline_and_ranking_use_same_sequential_full_coverage_loader(tmp_path):
    sequences = [
        [0, 0, 1, 10],
        [0, 0, 1, 11, 0, 1, 1, 12],
        [0, 0, 1, 13, 0, 1, 1, 14, 0, 2, 1, 15],
    ]
    path = tmp_path / "sequences.pkl"; path.write_bytes(pickle.dumps(sequences))
    baseline = make_data("spring", 223, str(path), batch_size=2)
    ranking = make_data("spring", 223, str(path), batch_size=2)
    assert isinstance(baseline.sampler, SequentialSampler)
    assert isinstance(ranking.sampler, SequentialSampler)
    assert list(iter(baseline.sampler)) == list(iter(ranking.sampler)) == [0, 1, 2]
    baseline_batches = [batch[0].clone() for batch in baseline]
    ranking_batches = [batch[0].clone() for batch in ranking]
    assert all(torch.equal(left, right) for left, right in zip(baseline_batches, ranking_batches))


def test_unit_weights_match_batch_loss_gradient_and_parameter_update():
    torch.manual_seed(3)
    first = torch.nn.Linear(3, 4, bias=False)
    second = torch.nn.Linear(3, 4, bias=False)
    second.load_state_dict(first.state_dict())
    x = torch.randn(2, 3)
    target = torch.tensor([1, 2])
    mask = torch.ones(2, 1)
    optimizers = [torch.optim.SGD(first.parameters(), lr=0.1), torch.optim.SGD(second.parameters(), lr=0.1)]
    losses = []
    for model, optimizer, weight in ((first, optimizers[0], None), (second, optimizers[1], [1.0, 1.0])):
        token_loss = torch.nn.functional.cross_entropy(model(x), target, reduction="none").reshape(2, 1)
        loss = masked_weighted_loss(token_loss, mask, weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.detach())
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(first.weight, second.weight)


def test_nonuniform_loss_still_uses_every_sample_once():
    token_loss = torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)
    loss = masked_weighted_loss(token_loss, torch.ones_like(token_loss), [0.9, 1.0, 1.1])
    loss.backward()
    assert torch.all(token_loss.grad > 0)
    assert torch.isclose(token_loss.grad.sum(), torch.tensor(1.0))
