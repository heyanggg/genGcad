import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('formal_saving',ROOT/'rasr_research/save_formal.py')
saving=importlib.util.module_from_spec(spec);spec.loader.exec_module(saving)


def test_registry_is_the_two_requested_scenarios_not_two_us_variants():
    records=json.loads((ROOT/'rasr_research/formal_records/registry.json').read_text(encoding='utf-8'))
    first_two=records['formal_groups'][:2]
    assert [r['scenario'] for r in first_two]==['SP-ST','US-ST']
    assert [round(r['metrics']['f1_score'],8) for r in first_two]==[.98201936,.95974543]
    assert all(r['status']=='formal' and r['f1_gain_percentage_points']>0 for r in first_two)
    assert first_two[1]['common_tuned_Gen_GCAD_metrics']['f1_score']==.9592875318066157
    assert 'previously exposed test' in first_two[1]['evidence_type']


def test_completed_fr_keeps_original_failure_and_tuned_formal_result():
    registry=saving.read(ROOT/'rasr_research/formal_records/registry.json')
    formal=next(r for r in registry['formal_groups'] if r['id']=='FR-ST-B3-tuned-seed2024')
    original=next(r for r in registry['supporting_groups'] if r['id']=='FR-ST-B3-original-seed2024')
    assert formal['metrics']['f1_score']==.9887640449438202
    assert formal['metrics']['FP']==2 and formal['metrics']['FN']==0
    assert formal['common_tuned_Gen_GCAD_metrics']['f1_score']==.9617486338797814
    assert original['metrics']['f1_score']==.7428571428571429
    assert original['status']=='supporting-not-above-baseline'


def test_portable_backup_excludes_raw_datasets_and_credentials(tmp_path):
    for relative in ('runtime/IoT_data/fr/spring/split_test.pkl','runtime/attack/labels.pkl',
                     'I.pkl','source_for_ssc.pkl','auth.json','.env'):
        file=tmp_path/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'fixture')
        assert not saving.safe_artifact(file)
    for relative in ('runtime/anomaly_runs/train.pkl','runtime/anomaly_runs/detector.pth',
                     'codex/call-00/response.txt','all_candidates.json'):
        file=tmp_path/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'fixture')
        assert saving.safe_artifact(file)


def test_lower_next_result_is_saved_as_supporting_not_a_formal_success(tmp_path,monkeypatch):
    records=tmp_path/'records';records.mkdir()
    monkeypatch.setattr(saving,'ROOT',tmp_path);monkeypatch.setattr(saving,'RECORDS',records)
    saving.write(records/'registry.json',{'formal_groups':[{'id':'existing-SP'},{'id':'existing-US'}]})
    formal=tmp_path/'formal'
    saving.write(formal/'fr/seed-2024/B3/result.json',{'actual':'lower real result'})
    saving.write(formal/'single-b3-comparison-fr.json',{'status':'completed',
        'comparison_baseline_metrics':{'f1_score':.96},'B3_metrics':{'f1_score':.90},
        'threshold_protocol':'original FR p95.5'})
    args=SimpleNamespace(formal=formal,dataset='fr',tuning=None)
    saving.register_next(args);saving.register_next(args)
    registry=saving.read(records/'registry.json')
    assert len(registry['formal_groups'])==2
    assert len(registry['supporting_groups'])==1
    assert registry['supporting_groups'][0]['metrics']['f1_score']==.90
    assert (records/'fr-st-original-comparison.json').exists()
