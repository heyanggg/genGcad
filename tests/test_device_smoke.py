import pytest
import torch

from SmartGen.gcad_source.mixer_predictor import SourceGCADMixer


def test_cpu_smoke():
    model = SourceGCADMixer(channels=3, history_length=2, hidden_size=4, num_layers=1, dropout=0.0)
    assert model(torch.zeros(2, 2, 3)).shape == (2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable on this host")
def test_cuda_smoke_when_available():
    model = SourceGCADMixer(channels=3, history_length=2, hidden_size=4, num_layers=1, dropout=0.0).cuda()
    assert model(torch.zeros(2, 2, 3, device="cuda")).is_cuda
