"""One explicitly reviewed quota-failure resume, not an automatic retry loop."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'rasr_research/runs/formal-us-mt-20261125/us/seed-2024/B3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert RUN.resolve().is_relative_to(ROOT.resolve())
    receipt=RUN/'explicit-resume-20261125.json'
    if receipt.exists():
        print('Review already applied; no files moved or requests started')
        return
    policy=read(ROOT/'rasr_research/generation-policy.json')
    assert policy['hold'] and policy['allowed_runs']==['us/seed-2024/B3']
    assert policy['allowed_call_ids']==[f'call-{i:02d}' for i in range(26)]
    preserved={}
    for index in range(26):
        directory=RUN/f'codex/call-{index:02d}'
        assert sha(directory/'prompt.txt')==read(directory/'request.json')['prompt_sha256']
        if index<3:
            assert read(directory/'exit.json')['returncode']==0 and (directory/'response.txt').exists()
            preserved[directory.name]=sha(directory/'response.txt')
        elif index<6:
            assert read(directory/'exit.json')['returncode']==1
            assert 'Your workspace is out of credits' in read(directory/'worker-error.json')['error']
            assert not (directory/'response.txt').exists() and (directory/'stdout.txt').stat().st_size==0
        else:
            assert not any((directory/name).exists() for name in ('worker-claim.json','worker-error.json','response.txt'))
    moves=[]
    # Move only verified diagnostic FILES into each same-slot recoverable folder.
    # Never remove prompts, requests, successful outputs or any original dataset.
    for index in (3,4,5):
        directory=RUN/f'codex/call-{index:02d}'
        archive=directory/'prior-failed-attempt-before-explicit-resume-20261125'
        assert not archive.exists();archive.mkdir()
        for path in list(directory.iterdir()):
            if path.is_file() and path.name not in ('request.json','prompt.txt','input_audit.json'):
                destination=archive/path.name
                moves.append({'old':str(path.relative_to(ROOT)),'saved':str(destination.relative_to(ROOT)),'sha256':sha(path)})
                shutil.move(str(path),str(destination))
    halt=RUN/'generation-halt.json'
    archived_halt=RUN/'prior-generation-halt-before-explicit-resume-20261125.json'
    assert halt.exists() and not archived_halt.exists()
    moves.append({'old':str(halt.relative_to(ROOT)),'saved':str(archived_halt.relative_to(ROOT)),'sha256':sha(halt)})
    shutil.move(str(halt),str(archived_halt))
    record={'authority':'Explicit human message: 恢复任务; manual review of three quota-only failures',
        'completed_slots_reused':preserved,'reviewed_failed_slots':['call-03','call-04','call-05'],
        'unstarted_slots':[f'call-{i:02d}' for i in range(6,26)],
        'max_new_attempts':23,'total_unique_slots':26,'prior_failed_attempts':3,
        'expected_total_attempts_if_completed':29,'automatic_retries':False,
        'prompts_model_and_algorithms_changed':False,'preserved_files':moves}
    receipt.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'successful_slots_reused':3,'prior_failed_attempts_preserved':3,'remaining_slots':23,'new_calls_by_review':0}))


if __name__=='__main__':main()
