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


def test_completed_fr_daytime_keeps_original_decline_and_common_tuned_control():
    records=saving.read(ROOT/'rasr_research/formal_records/registry.json')
    tuned=next(r for r in records['formal_groups'] if r['id']=='FR-DT-B3-tuned-seed2024')
    negative=next(r for r in records['supporting_groups'] if r['id']=='FR-DT-B3-original-seed2024')
    assert negative['metrics']['f1_score']==.8964218455743879
    assert negative['metrics']['FP']==220 and negative['metrics']['FN']==0
    assert tuned['metrics']['f1_score']==.9916666666666667
    assert tuned['metrics']['FP']==16 and tuned['metrics']['FN']==0
    assert tuned['common_tuned_Gen_GCAD_metrics']['f1_score']==.9333333333333333
    assert tuned['Gen_GCAD_reference_metrics']['f1_score']==.9719244512506381
    assert tuned['protocol']['epochs']==5 and tuned['protocol']['score']=='mean'
    assert tuned['gcad_comparison_state']['archived_mode'] is None
    lock=saving.read(ROOT/'rasr_research/formal_records/fr-dt-tuning-lock-before-confirmation.json')
    confirmation=saving.read(ROOT/'rasr_research/formal_records/fr-dt-tuned-confirmation.json')
    assert lock['selection_sha256']==confirmation['selection_sha256']
    assert lock['parameters']==confirmation['selection']['parameters']


def test_fr_daytime_volume_wording_correction_preserves_every_metric_and_lock():
    folder=ROOT/'rasr_research/formal_records'
    note=saving.read(folder/'fr-dt-summary-wording-note.json')
    before=saving.read(folder/note['initial_report']);after=saving.read(folder/note['corrected_report'])
    assert saving.sha(folder/note['initial_report'])==note['initial_sha256']
    assert saving.sha(folder/note['corrected_report'])==note['corrected_sha256']
    assert {k:v for k,v in before.items() if k!='limitations'}=={k:v for k,v in after.items() if k!='limitations'}
    assert sum(a!=b for a,b in zip(before['limitations'],after['limitations']))==1
    assert saving.sha(folder/'fr-dt-tuned-confirmation.json')==note['confirmation_sha256_unchanged']
    assert saving.sha(folder/'fr-dt-tuned-selection.json')==note['selection_sha256_unchanged']


def test_legacy_above_historical_formal_group_keeps_unmatched_protocol_caveat(tmp_path,monkeypatch):
    records=tmp_path/'records';records.mkdir()
    monkeypatch.setattr(saving,'ROOT',tmp_path);monkeypatch.setattr(saving,'RECORDS',records)
    saving.write(records/'registry.json',{'formal_groups':[]})
    formal=tmp_path/'formal'
    saving.write(formal/'sp/seed-2024/B3/result.json',{'actual':'fixture'})
    saving.write(formal/'single-b3-comparison.json',{'status':'completed','scenario':'SP-MT',
        'comparison_baseline_metrics':{'f1_score':.91},'B3_metrics':{'f1_score':.92},
        'matched_protocol_baseline':{'metrics':{'f1_score':.95}},
        'historical_protocol_caveat':'Historical detector split unavailable',
        'threshold_protocol':'p99 new split'})
    saving.register_next(SimpleNamespace(formal=formal,dataset='sp',tuning=None))
    entry=saving.read(records/'registry.json')['formal_groups'][0]
    assert entry['Gen_GCAD_reference_metrics']['f1_score']==.91
    assert entry['matched_protocol_Gen_GCAD_metrics']['f1_score']==.95
    assert entry['B3_minus_matched_protocol_f1_percentage_points']<0
    assert 'not evidence' in entry['formal_label_caveat']


def test_completed_sp_multiple_preserves_all_three_controls_and_no_unneeded_tuning():
    folder=ROOT/'rasr_research/formal_records'
    registry=saving.read(folder/'registry.json')
    entry=next(r for r in registry['formal_groups'] if r['id']=='SP-MT-B3-original-seed2024')
    assert entry['metrics']['f1_score']==.9404761904761905
    assert entry['metrics']['FP']==20 and entry['metrics']['FN']==0
    assert entry['Gen_GCAD_reference_metrics']['f1_score']==.9132947976878613
    assert entry['matched_protocol_Gen_GCAD_metrics']['f1_score']==0
    assert 'NOT an apples-to-apples' in entry['historical_protocol_caveat']
    comparison=saving.read(folder/'sp-mt-original-comparison.json')
    assert comparison['generation']['valid_responses']==8
    assert comparison['generation']['new_baseline_calls']==0
    assert comparison['prompt']['actual_examples']==176 and comparison['prompt']['J_examples']==119
    assert comparison['data_volumes']['B3_TOF_final']==71
    note=saving.read(folder/'sp-mt-completion-note.json')
    assert note['tuning']['triggered'] is False
    assert note['result_sha256']==entry['result_sha256']


def test_original_us_multiple_negative_is_preserved_not_mislabeled_formal():
    folder=ROOT/'rasr_research/formal_records'
    registry=saving.read(folder/'registry.json')
    entry=next(r for r in registry['supporting_groups'] if r['id']=='US-MT-B3-original-seed2024')
    assert entry['metrics']['f1_score']==0 and entry['metrics']['FN']==3209
    assert entry['Gen_GCAD_reference_metrics']['f1_score']==.9417461482024945
    assert entry['matched_protocol_Gen_GCAD_metrics']['f1_score']==.896995108315863
    assert 'Not a formal' in entry['formal_label_caveat']
    note=saving.read(folder/'us-mt-negative-label-wording-note.json')
    assert note['original_result_sha256_unchanged']==entry['result_sha256']
    comparison=saving.read(folder/'us-mt-original-comparison.json')
    assert comparison['generation']['valid_responses']==26
    assert comparison['generation']['total_CLI_attempts_including_saved_failures']==29
    assert comparison['generation']['saved_prior_failed_CLI_attempts']==3
    assert comparison['prompt']['actual_examples']==689 and comparison['prompt']['J_examples']==505
