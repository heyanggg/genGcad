"""Offline budget guards: no real model requests in these tests."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_codex_worker',ROOT/'rasr_research/codex_worker.py')
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def policy_at(root, hold=False):
    policy={'queue_root':'.','allowed_runs':['sp/seed-2024/B3'],
            'max_additional_calls':16,'max_workers':1,'hold':hold}
    (root/'generation-policy.json').write_text(json.dumps(policy))
    return policy


def test_child_root_cannot_bypass_single_b3_policy(tmp_path):
    policy=policy_at(tmp_path)
    child=tmp_path/'sp/seed-2024/B3'
    anchor,found=worker.generation_policy(child)
    assert anchor==tmp_path
    assert found==policy
    assert worker.request_allowed(child/'codex/call-00/request.json',anchor,found)
    with_slots=dict(found,allowed_call_ids=['call-00'])
    assert not worker.request_allowed(child/'codex/call-01/request.json',anchor,with_slots)
    for other in ('sp/seed-2024/B0','sp/seed-2024/B1','us/seed-2024/B3','sp/seed-2025/B3'):
        assert not worker.request_allowed(tmp_path/other/'codex/call-00/request.json',anchor,found)


def test_hold_denies_even_authorized_b3_and_makes_zero_client_calls(tmp_path,monkeypatch):
    policy=policy_at(tmp_path,hold=True)
    assert not worker.request_allowed(tmp_path/'sp/seed-2024/B3/codex/call-00/request.json',tmp_path,policy)
    def forbidden_client(**kwargs):
        raise AssertionError('policy hold must never instantiate a real client')
    monkeypatch.setattr(worker,'CodexClient',forbidden_client)
    monkeypatch.setattr(sys,'argv',['worker','--root',str(tmp_path),'--codex-bin','unused'])
    worker.main()


def test_first_failure_halts_queue_without_trying_next_request(tmp_path,monkeypatch):
    policy_at(tmp_path)
    for number in range(2):
        directory=tmp_path/f'sp/seed-2024/B3/codex/call-{number:02d}'
        directory.mkdir(parents=True)
        (directory/'prompt.txt').write_text('offline fixture')
        (directory/'request.json').write_text(json.dumps({'model':'gpt-5.6-sol',
            'reasoning_effort':'none','prompt_sha256':hashlib.sha256(b'offline fixture').hexdigest()}))
    attempts=[]
    def failed_client(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError('offline quota-error fixture')
    monkeypatch.setattr(worker,'CodexClient',failed_client)
    monkeypatch.setattr(sys,'argv',['worker','--root',str(tmp_path),'--codex-bin','unused','--workers','6'])
    worker.main()
    assert len(attempts)==1
    assert (tmp_path/'generation-halt.json').exists()
    assert not (tmp_path/'sp/seed-2024/B3/codex/call-01/worker-claim.json').exists()
