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


def test_daytime_result_cannot_overwrite_seasonal_us_records(tmp_path,monkeypatch):
    records=tmp_path/'records';records.mkdir()
    monkeypatch.setattr(saving,'ROOT',tmp_path);monkeypatch.setattr(saving,'RECORDS',records)
    saving.write(records/'registry.json',{'formal_groups':[{'id':'US-ST-B3-tuned-seed2024'}]})
    formal=tmp_path/'formal'
    saving.write(formal/'us/seed-2024/B3/result.json',{'actual':'daytime result'})
    saving.write(formal/'single-b3-comparison-us.json',{'status':'completed','scenario':'US-DT',
        'comparison_baseline_metrics':{'f1_score':.90},'B3_metrics':{'f1_score':.91},
        'threshold_protocol':'original US night p93'})
    args=SimpleNamespace(formal=formal,dataset='us',tuning=None)
    saving.register_next(args);saving.register_next(args)
    registry=saving.read(records/'registry.json')
    assert [r['id'] for r in registry['formal_groups']]==['US-ST-B3-tuned-seed2024','US-DT-B3-original-seed2024']
    assert (records/'us-dt-original-comparison.json').exists()
    assert not (records/'us-st-original-comparison.json').exists()


def test_three_prior_release_backups_are_verified_by_remote_sha_and_size():
    for name in ('backup-receipt.json','fr-st-original-backup-receipt.json','fr-st-tuned-backup-receipt.json'):
        receipt=saving.read(ROOT/'rasr_research/formal_records'/name)
        assert receipt['github_uploaded'] is True
        assert receipt['github_asset_size']==receipt['bytes']
        assert receipt['github_digest']=='sha256:'+receipt['sha256']


def test_us_daytime_formal_result_keeps_original_protocol_and_residual_false_negatives():
    registry=saving.read(ROOT/'rasr_research/formal_records/registry.json')
    result=next(r for r in registry['formal_groups'] if r['id']=='US-DT-B3-original-seed2024')
    assert result['metrics']['f1_score']==.9230055658627088
    assert result['metrics']['FP']==0 and result['metrics']['FN']==498
    assert result['Gen_GCAD_reference_metrics']['f1_score']==.9035871045860452
    assert 'p93' in result['protocol']
    assert 'parameters' not in result
    diagnostic=saving.read(ROOT/'rasr_research/formal_records/us-dt-posthoc-diagnostics.json')
    assert diagnostic['conditional_tuning_triggered'] is False
    assert diagnostic['already_completed_test_score_diagnostics']['used_for_parameter_selection'] is False


def test_auto_disabled_gcad_caveat_survives_sp_formal_registration(tmp_path,monkeypatch):
    records=tmp_path/'records';records.mkdir()
    monkeypatch.setattr(saving,'ROOT',tmp_path);monkeypatch.setattr(saving,'RECORDS',records)
    saving.write(records/'registry.json',{'formal_groups':[]})
    formal=tmp_path/'formal'
    saving.write(formal/'sp/seed-2024/B3/result.json',{'actual':'fixture only'})
    state={'guidance_enabled':False,'archived_mode':'auto','interpretation':'No active guidance'}
    saving.write(formal/'single-b3-comparison.json',{'status':'completed','scenario':'SP-DT',
        'comparison_baseline_metrics':{'f1_score':.98},'B3_metrics':{'f1_score':.99},
        'threshold_protocol':'original p95','gcad_comparison_state':state})
    saving.register_next(SimpleNamespace(formal=formal,dataset='sp',tuning=None))
    entry=saving.read(records/'registry.json')['formal_groups'][0]
    assert entry['id']=='SP-DT-B3-original-seed2024'
    assert entry['gcad_comparison_state']==state
    assert 'not label this an active-GCAD' in entry['baseline_caveat']


def test_completed_sp_daytime_result_retains_actual_metrics_and_protocol_limits():
    records=saving.read(ROOT/'rasr_research/formal_records/registry.json')
    entry=next(r for r in records['formal_groups'] if r['id']=='SP-DT-B3-original-seed2024')
    assert entry['metrics']['f1_score']==.9912841371295759
    assert entry['metrics']['FP']==30 and entry['metrics']['FN']==0
    assert entry['Gen_GCAD_reference_metrics']['f1_score']==.9844200807847664
    assert entry['gcad_comparison_state']['guidance_enabled'] is False
    note=saving.read(ROOT/'rasr_research/formal_records/sp-dt-completion-note.json')
    assert note['conditional_tuning_triggered'] is False
    assert note['actual_new_codex_calls']==13 and note['new_B0_calls']==0
    assert note['B3_volumes']['detector_train']==41
    assert note['historical_volumes']['detector_train']==40


def test_legacy_disabled_gcad_registration_does_not_invent_auto_mode(tmp_path,monkeypatch):
    records=tmp_path/'records';records.mkdir()
    monkeypatch.setattr(saving,'ROOT',tmp_path);monkeypatch.setattr(saving,'RECORDS',records)
    saving.write(records/'registry.json',{'formal_groups':[]})
    formal=tmp_path/'formal'
    saving.write(formal/'fr/seed-2024/B3/result.json',{'actual':'fixture only'})
    state={'guidance_enabled':False,'archived_mode':None,'mode_field_missing_in_legacy_config':True}
    saving.write(formal/'single-b3-comparison-fr.json',{'status':'completed','scenario':'FR-DT',
        'comparison_baseline_metrics':{'f1_score':.97},'B3_metrics':{'f1_score':.98},
        'threshold_protocol':'original p95','gcad_comparison_state':state})
    saving.register_next(SimpleNamespace(formal=formal,dataset='fr',tuning=None))
    entry=saving.read(records/'registry.json')['formal_groups'][0]
    assert entry['gcad_comparison_state']['archived_mode'] is None
    assert 'no mode flag recorded' in entry['baseline_caveat']
    assert 'auto' not in entry['baseline_caveat']
