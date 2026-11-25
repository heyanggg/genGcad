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


def test_legacy_fr_tof_metadata_uses_only_actual_archived_counts(tmp_path):
    folder=tmp_path/'artifacts/tof';folder.mkdir(parents=True)
    formal.atomic_pickle_dump(folder/'fr_seq.pkl',[[1],[2],[3]])
    formal.atomic_pickle_dump(folder/'fr_seq_filter.pkl',[[1]])
    formal.atomic_pickle_dump(folder/'fr_seq_filter_true.pkl',[[1],[2]])
    counts=formal.archived_tof(tmp_path,{})
    assert (counts['input_count'],counts['first_pass_count'],counts['final_count'])==(3,1,2)
    assert formal.vocabulary_size('fr')==223


def test_new_fr_embedding_does_not_invalidate_old_frozen_sp_us_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(formal,'ROOT',tmp_path)
    rasr=tmp_path/'SmartGen/rasr.py';rasr.parent.mkdir();rasr.write_text('unchanged algorithm')
    embed=tmp_path/'relation_research/embeddings/rasr-raw-v1';embed.mkdir(parents=True)
    for ds in ('sp','us'):
        (embed/f'{ds}_embeddings.json').write_text(ds)
        prepared=formal.prepared_path(ds);prepared.mkdir(parents=True)
        for name in ('relations.npz','B1_J.pkl','B3_J.pkl','I.pkl'):(prepared/name).write_bytes(b'unchanged')
    before=formal.frozen_files()
    (embed/'fr_embeddings.json').write_text('new independent scene')
    assert formal.frozen_files()==before


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
    assert policy['allowed_runs']==['sp/seed-2024/B3']
    assert policy['queue_root']=='runs/formal-sp-mt-20261125'
    assert policy['max_additional_calls']==8
    assert policy['max_workers']==3
    assert policy['allowed_call_ids']==[f'call-{i:02d}' for i in range(8)]
    assert isinstance(policy['hold'],bool)
    if not policy['hold']:
        assert 'Explicit human reply' in policy['resume_authority']
    assert policy['automatic_retries'] is False
    source=(ROOT/'rasr_research/formal_experiment.py').read_text()
    assert "choices=['audit','prepare-single','evaluate-single','score-historical','wait-single','recover-stdout','status']" in source
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


@pytest.mark.parametrize('dataset,vocab',[('sp',235),('us',269),('fr',223)])
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


def test_new_environment_paths_and_inputs_cannot_reuse_seasonal_source():
    assert formal.prepared_path('us','daytime')!=formal.prepared_path('us')
    assert formal.test_relatives('us','night')==('IoT_data/us/night/split_test.pkl','attack/us/labeled_us_night_attack_time.pkl')
    protocol={'datasets':{'us':{'reference_config':{'target_environment':'night'}}}}
    assert formal.target_environment(protocol,'us')=='night'
    assert formal.scenario(protocol,'us')=='US-DT'


def test_archived_auto_disabled_gcad_is_preserved_and_not_claimed_active():
    state=formal.preserved_gcad_state({'gcad_mode':'auto'},
        {'status':'disabled','guidance_enabled':False,'relationship_count':0})
    assert state['guidance_enabled'] is False
    assert 'NOT an active-GCAD' in state['interpretation']
    with pytest.raises(ValueError):
        formal.preserved_gcad_state({'gcad_mode':'require'},
            {'status':'disabled','guidance_enabled':False,'relationship_count':0})
    assert formal.preserved_gcad_state({'gcad_mode':'require'},
        {'status':'ready','guidance_enabled':True})['guidance_enabled'] is True


def test_legacy_disabled_gcad_requires_matching_empty_archived_artifact_not_invented_mode():
    config={};run={'status':'disabled','relationship_count':0,'artifact_fingerprint':'frozen-proof'}
    with pytest.raises(ValueError):formal.preserved_gcad_state(config,run)
    artifact={'status':'disabled','lagged_behavior_relations':[],
        'artifact_fingerprint':'frozen-proof','disabled_reason':'no_stable_edge'}
    state=formal.preserved_gcad_state(config,run,artifact)
    assert state['archived_mode'] is None and state['guidance_enabled'] is False
    assert state['mode_field_missing_in_legacy_config'] is True
    assert config=={} and 'guidance_enabled' not in run
    with pytest.raises(ValueError):
        formal.preserved_gcad_state(config,run,dict(artifact,artifact_fingerprint='wrong'))
    with pytest.raises(ValueError):
        formal.preserved_gcad_state(config,run,dict(artifact,lagged_behavior_relations=[{'edge':'unexpected'}]))


def test_legacy_reevaluation_requires_exact_saved_final_pool_and_immutable_receipt(tmp_path):
    archive=tmp_path/'archive';tof=archive/'artifacts/tof';tof.mkdir(parents=True)
    final=tof/'legacy_seq_filter_true.pkl';formal.atomic_pickle_dump(final,[[1],[2],[3]])
    folder=tmp_path/'reevaluation';folder.mkdir()
    formal.atomic_pickle_dump(folder/'train.pkl',[[3],[1]])
    formal.atomic_pickle_dump(folder/'validation.pkl',[[2]])
    original={'generated_sequence_count':3,'training_sequence_count':3,'validation_sequence_count':3}
    saved={'new_codex_calls':0,'TOF_rerun':False,'historical_split_recovered':False,
        'original_record':original,'input':str(final),'input_sha256':formal.sha(final),
        'detector_dir':str(folder),'detection':{'training_sequence_count':2,'validation_sequence_count':1},
        'files_sha256':{str(p):formal.sha(p) for p in (final,folder/'train.pkl',folder/'validation.pkl')}}
    receipt=tmp_path/'result.json';formal.atomic_write_json(receipt,saved)
    assert formal.verified_baseline_reevaluation(receipt,archive,original)==saved
    assert formal.baseline_detector_dir({'archive':str(archive),'baseline_reevaluation':saved})==folder
    formal.atomic_pickle_dump(folder/'validation.pkl',[[99]])
    with pytest.raises(AssertionError):formal.verified_baseline_reevaluation(receipt,archive,original)
