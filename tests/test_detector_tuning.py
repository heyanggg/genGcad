import copy
import importlib.util
import inspect
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('detector_tuning',ROOT/'rasr_research/tune_detector.py')
tune=importlib.util.module_from_spec(spec);spec.loader.exec_module(tune)


def row(*actions):
    return [v for a in actions for v in (0,0,1,a)]


def test_covered_validation_has_no_duplicate_groups_or_unseen_normal_actions():
    rows=[row(1,2),row(1,2),row(2,3),row(3,1),row(1,3),row(7)]
    before=copy.deepcopy(rows)
    training,validation=tune.coverage_split(rows,2024,ratio=.3)
    assert not {tuple(r) for r in training}&{tuple(r) for r in validation}
    assert {a for r in validation for a in tune.tokens(r)}<={a for r in training for a in tune.tokens(r)}
    assert row(7) in training
    assert rows==before
    assert tune.coverage_split(rows,2024,ratio=.3)==(training,validation)


def test_proxies_use_unseen_legal_dictionary_actions_not_a_named_attack():
    rows=[row(1,2),row(2,3)]
    proxies=tune.proxy_rows(rows,rows,[1,2,3,4,5],42)
    assert len(proxies)==6
    for i,p in enumerate(proxies):
        original=rows[i//3]
        different=[(a,b) for a,b in zip(tune.tokens(original),tune.tokens(p)) if a!=b]
        assert len(different)==1 and different[0][1] in (4,5)
    assert proxies==tune.proxy_rows(rows,rows,[1,2,3,4,5],42)


def test_short_normal_views_are_derived_only_from_the_held_out_row():
    rows=[row(1,2,3)]
    views=tune.normal_views(rows)
    assert row(1) in views and row(2) in views and row(3) in views
    assert row(1,2) in views and row(1,2,3) in views
    assert all(set(tune.tokens(r))<={1,2,3} for r in views)
    assert rows==[row(1,2,3)]


def test_scoring_modes_ignore_padding_and_use_no_new_loss():
    class Fixed(torch.nn.Module):
        def forward(self,src,src_key_padding_mask=None):
            return torch.arange(10,dtype=torch.float32).expand(len(src),10,10)
    scores=tune.score(Fixed(),[row(2),row(2,4)],10,torch.device('cpu'))
    nll=-torch.log_softmax(torch.arange(10,dtype=torch.float32),0).numpy()
    assert np.allclose(scores[0],[nll[2]]*3)
    assert np.allclose(scores[1],[(nll[2]+nll[4])/2,max(nll[2],nll[4]),(nll[2]+nll[4])/2])


def test_fit_and_model_selection_do_not_read_target_test_files():
    source=inspect.getsource(tune.fit)+inspect.getsource(tune.inputs)
    assert 'split_test' not in source and 'labeled_' not in source
    assert 'new_codex_calls' in source
    assert 'selection.json' in inspect.getsource(tune.confirm)
    assert 'Confirmation already exists' in inspect.getsource(tune.confirm)


def test_real_transformer_batched_scores_match_single_row_scores():
    # Exercise mixed padding and the 64-row scoring boundary, with no checkpoints
    # or target data. Batch inference must not silently change the detector rule.
    torch.manual_seed(2024)
    model=tune.TransformerAutoencoder(10,d_model=16,nhead=2,
        num_encoder_layers=1,num_decoder_layers=1)
    rows=[row(1),row(2,3),row(3,4,5)]*22
    batched=tune.score(model,rows,10,torch.device('cpu'))
    unique=np.concatenate([tune.score(model,[r],10,torch.device('cpu')) for r in rows[:3]])
    assert np.allclose(batched,np.tile(unique,(22,1)),atol=1e-5,rtol=1e-5)
