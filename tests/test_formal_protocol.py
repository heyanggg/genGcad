"""Formal protocol integrity and exact legacy compression/conversion semantics."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SmartGen'))
spec=importlib.util.spec_from_file_location('formal_experiment',ROOT/'rasr_research/formal_experiment.py')
formal=importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


def test_original_greedy_matches_legacy_matrix_selection():
    rng=np.random.RandomState(2024)
    rows=rng.randn(100,16).astype(np.float32)
    rows[20]=rows[0]; rows[60]=rows[5]
    similarity=cosine_similarity(rows)
    removed=set(); expected=[]
    for i in range(len(rows)):
        if i not in removed:
            expected.append(i)
            removed.update(j for j in range(i+1,len(rows)) if similarity[i,j]>.8)
    assert formal.original_greedy(rows,.8)==expected


def test_archived_prompt_roundtrip_preserves_template_byte_for_byte():
    text='prefix '+formal.MARKER+"[['Monday', '(0~3)', 'Light', 'Light:switch on']]"+formal.END+'unchanged GSS and GCAD'
    before,examples,after=formal.unpack_prompt(text)
    assert before+str(examples)+after==text


def test_numeric_trace_retains_existing_alias_and_unknown_quad_filter():
    maps=[{'Monday':0},{'(0~3)':0},{'Light':1},{'Light:switch on':10}]
    assert formal.numeric_sequence(['Monday','(0~3)','Light','switch on'],maps)==[0,0,1,10]
    assert formal.numeric_sequence(['Monday','(0~3)','Light','Light:switch on'],maps)==[0,0,1,10]
    assert formal.numeric_sequence(['Monday','(0~3)','Unknown','switch on'],maps)==[]


def test_historical_constants_preserved_without_new_regenerator():
    assert formal.SEEDS==(2024,2025,2026)
    assert formal.ARMS==('B0','B1','B3')
    source=(ROOT/'rasr_research/formal_experiment.py').read_text()
    assert 'RelationGenerator' not in source
    assert 'model.train()' in source
    assert 'model.eval()' not in source


def test_short_ssc_output_keeps_call_budget_without_refill_or_duplicates():
    templates=[{'day':3,'examples':30},{'day':3,'examples':3}]
    assert formal.category_partition({3:['a','b']},templates)==[['a'],['b']]
    assert formal.category_partition({3:['a']},templates)==[['a'],[]]
    assert formal.category_partition({3:[]},templates)==[[],[]]


def test_full_output_preserves_historical_partition_and_prefix():
    templates=[{'day':3,'examples':30},{'day':3,'examples':3}]
    selected=formal.category_partition({3:list(range(33))},templates)
    assert selected==[list(range(30)),list(range(30,33))]


def test_active_authority_is_single_b3_not_the_retired_multi_arm_schedule():
    policy=json.loads((ROOT/'rasr_research/generation-policy.json').read_text())
    assert policy['allowed_runs']==['us/seed-2024/B3']
    assert policy['max_additional_calls']==11
    assert policy['max_workers']==3
    assert policy['allowed_call_ids']==[f'call-{i:02d}' for i in range(11)]
    assert isinstance(policy['hold'],bool)
    if not policy['hold']:
        assert 'Explicit human reply' in policy['resume_authority']
    assert policy['automatic_retries'] is False
    source=(ROOT/'rasr_research/formal_experiment.py').read_text()
    assert "choices=['audit','prepare-single','evaluate-single','score-historical','wait-single','recover-stdout']" in source
    assert "seeds=(2024,)" in source


def test_us_reports_are_isolated_and_sp_existing_filenames_stay_unchanged(tmp_path):
    args=SimpleNamespace(output=tmp_path,dataset='us')
    assert formal.single_dataset(args)=='us'
    assert formal.single_file(args,'us','single-b3-comparison')==tmp_path/'single-b3-comparison-us.json'
    assert formal.single_file(args,'sp','single-b3-comparison')==tmp_path/'single-b3-comparison.json'
    assert formal.single_file(args,'us','active-scope')==tmp_path/'active-scope-us.json'
    assert formal.single_dataset(SimpleNamespace())=='sp'


def test_recover_real_stdout_without_inventing_exit_status_or_repeating_call(tmp_path):
    prompt=tmp_path/'prompt.txt'; prompt.write_text('same original prompt')
    block="<seq [['Monday','(0~3)','Light','Light:switch on']] seq>"
    (tmp_path/'stdout.txt').write_text(block)
    (tmp_path/'request.json').write_text(json.dumps({'prompt_sha256':formal.sha(prompt),
        'model':'gpt-5.6-sol','reasoning_effort':'none'}))
    (tmp_path/'worker-error.json').write_text(json.dumps({'type':'TypeError','error':'data must be str, not NoneType'}))
    assert formal.recover_saved_response(tmp_path)
    receipt=formal.verify_response_receipt(tmp_path)
    assert receipt['returncode'] is None
    assert receipt['new_model_calls']==0
    assert not formal.recover_saved_response(tmp_path)
    (tmp_path/'stdout.txt').write_text('tampered')
    with pytest.raises(AssertionError):formal.verify_response_receipt(tmp_path)


@pytest.mark.parametrize('dataset,vocab',[('sp',235),('us',269)])
def test_historical_rescore_reuses_checkpoint_and_original_threshold_without_training(tmp_path,monkeypatch,dataset,vocab):
    archive=tmp_path/'archive'
    checkpoint=archive/'artifacts/anomaly_detection/detector.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'unchanged checkpoint fixture')
    source=tmp_path/'source'
    for relative in (f'SmartGen/attack/{dataset}/labeled_{dataset}_spring_attack_heater.pkl',f'SmartGen/IoT_data/{dataset}/spring/split_test.pkl'):
        path=source/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'read-only test fixture')
    names=('TP','TN','FP','FN','FPR','FNR','recall','precision','accuracy','f1_score')
    values=(2,3,0,0,0.,0.,1.,1.,1.,1.)
    original_threshold=1.2345
    protocol={'datasets':{dataset:{'archive':str(archive),'reference_metrics':{
        'detection_threshold':original_threshold,'metrics':dict(zip(names,values))}}}}
    invocations=[]
    def existing_evaluator(*args,**kwargs):
        assert args[1]==vocab
        assert args[4]==str(checkpoint)
        assert kwargs['threshold']==original_threshold
        assert kwargs['detailed'] is True
        invocations.append((args,kwargs))
        return {'threshold_metrics':values,'auroc':1.,'auprc':1.}
    monkeypatch.setattr(formal,'evaluate_detector',existing_evaluator)
    monkeypatch.setattr(formal,'single_report',lambda *args:None)
    args=SimpleNamespace(source=source,output=tmp_path/'output',dataset=dataset)
    formal.score_historical_checkpoint(args,protocol)
    formal.score_historical_checkpoint(args,protocol)  # reuses saved scores
    assert len(invocations)==1
    result=formal.read(args.output/f'historical-checkpoint-score/{dataset}/result.json')
    assert result['threshold_unchanged']
    assert result['matches_original_recorded_metrics']
    assert checkpoint.read_bytes()==b'unchanged checkpoint fixture'
