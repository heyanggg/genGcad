"""Immutable formal registry and portable experiment backup (standard library).

Human-designated formal status is separate from confirmatory/causal evidence.
Raw source/target datasets and credentials are never included in the backup.
Large weights/evidence are release assets, not oversized Git blobs.
"""
import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'rasr_research/formal_records'
BACKUPS=ROOT/'rasr_research/backups'
FORMAL=ROOT/'rasr_research/runs/formal-20261124'
TUNING=ROOT/'rasr_research/runs/tuning-20261124/covered-view-cv'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def register():
    RECORDS.mkdir(parents=True,exist_ok=True)
    sp=read(FORMAL/'single-b3-comparison.json')
    us=read(FORMAL/'single-b3-comparison-us.json')
    tuned=read(TUNING/'confirmation.json')
    entries=[{
        'id':'SP-ST-B3-original-seed2024','scenario':'SP-ST','status':'formal',
        'protocol':'Original formal detector: 15 epochs, mean reconstruction error, validation p95',
        'metrics':sp['B3_metrics'],'Gen_GCAD_reference_metrics':sp['comparison_baseline_metrics'],
        'result':str((FORMAL/'sp/seed-2024/B3/result.json').relative_to(ROOT)),
        'evidence_type':'Single-seed historical comparison, not stable causal proof',
        'comparison_report':'sp-st-comparison.json'}, {
        'id':'US-ST-B3-tuned-seed2024','scenario':'US-ST','status':'formal',
        'protocol':'Existing detector tuned on grouped generated-normal validation plus synthetic proxies; locked before final scoring',
        'parameters':tuned['selection']['parameters'],
        'metrics':tuned['arms']['B3']['metrics'],
        'Gen_GCAD_reference_metrics':us['comparison_baseline_metrics'],
        'common_tuned_Gen_GCAD_metrics':tuned['arms']['historical']['metrics'],
        'result':str((TUNING/'confirmation.json').relative_to(ROOT)),
        'evidence_type':'Exploratory on previously exposed test; changed detector protocol; not isolated RaSR gain',
        'comparison_report':'us-st-tuned-summary.json'}]
    registry_path=RECORDS/'registry.json'
    registry=read(registry_path) if registry_path.exists() else {
        'schema_version':1,'repository':'https://github.com/heyanggg/genGcad',
        'authority':'Human clarification: formal groups are SP-ST higher B3 and US-ST higher-F1 B3, not two US groups',
        'future_rule':'Retain every result; register completed B3 variants exceeding their corresponding historical Gen+GCAD F1 as formal, with exact protocol, tuning history, all attempts and limitations. Never use formal status as a statistical-significance claim.',
        'formal_groups':[]}
    for entry in entries:
        entry['f1_gain_percentage_points']=100*(entry['metrics']['f1_score']-entry['Gen_GCAD_reference_metrics']['f1_score'])
        assert entry['f1_gain_percentage_points']>0
        result=ROOT/entry['result'];entry['result_sha256']=sha(result)
        old=next((r for r in registry['formal_groups'] if r['id']==entry['id']),None)
        if old is not None and old!=entry:raise ValueError('Existing formal result changed: '+entry['id'])
        if old is None:registry['formal_groups'].append(entry)
    write(registry_path,registry)
    for original,name in [(FORMAL/'single-b3-comparison.json','sp-st-comparison.json'),
        (FORMAL/'single-b3-comparison-us.json','supporting-us-original-negative.json'),
        (TUNING/'summary.json','us-st-tuned-summary.json'),
        (TUNING/'confirmation.json','us-st-tuned-confirmation.json'),
        (TUNING/'selection.json','us-st-tuned-selection.json'),
        (TUNING/'protocol.json','us-st-tuning-protocol.json'),
        (TUNING.parent/'family-choice-before-test.json','us-st-family-choice-before-test.json')]:
        destination=RECORDS/name
        if destination.exists() and destination.read_bytes()!=original.read_bytes():
            raise ValueError('Saved evidence changed: '+name)
        if not destination.exists():shutil.copy2(original,destination)
    print('REGISTERED formal groups: '+', '.join(r['id'] for r in registry['formal_groups']),flush=True)


def safe_artifact(path):
    parts=path.parts
    return (path.is_file() and 'attack' not in parts and '__pycache__' not in parts
        and path.name not in {'split_test.pkl','trn.pkl','split_trn.pkl','source_for_ssc.pkl','I.pkl'}
        and path.suffix not in {'.pyc','.lock'} and path.name not in {'auth.json','credentials.json','.env'})


def register_next(args):
    directory=args.formal;comparison=read(directory/f'single-b3-comparison-{args.dataset}.json')
    if comparison['status']!='completed':raise ValueError('Cannot register an incomplete experiment')
    baseline=comparison['comparison_baseline_metrics']
    tuned=read(args.tuning/'confirmation.json') if args.tuning else None
    metrics=tuned['arms']['B3']['metrics'] if tuned else comparison['B3_metrics']
    variant='tuned' if tuned else 'original'
    result=args.tuning/'confirmation.json' if tuned else directory/f'{args.dataset}/seed-2024/B3/result.json'
    entry={'id':f'{args.dataset.upper()}-ST-B3-{variant}-seed2024','scenario':args.dataset.upper()+'-ST',
        'status':'formal' if metrics['f1_score']>baseline['f1_score'] else 'supporting-not-above-baseline',
        'metrics':metrics,'Gen_GCAD_reference_metrics':baseline,
        'f1_gain_percentage_points':100*(metrics['f1_score']-baseline['f1_score']),
        'result':result.relative_to(ROOT).as_posix(),'result_sha256':sha(result),
        'protocol':tuned['selection']['parameters'] if tuned else comparison['threshold_protocol'],
        'evidence_type':'Single-seed historical comparison; all attempts retained, no significance claim'}
    if tuned:
        entry['common_tuned_Gen_GCAD_metrics']=tuned['arms']['historical']['metrics']
        entry['evidence_type']=tuned['limitations']+' Detector tuning is not isolated RaSR gain.'
    filename=f'{args.dataset}-st-{variant}-comparison.json'
    destination=RECORDS/filename;original=args.tuning/'summary.json' if tuned else directory/f'single-b3-comparison-{args.dataset}.json'
    if destination.exists() and sha(destination)!=sha(original):raise ValueError('Saved result changed')
    if not destination.exists():shutil.copy2(original,destination)
    if tuned:
        for name in ('confirmation.json','selection.json','protocol.json'):
            target=RECORDS/f'{args.dataset}-st-tuned-{name}'
            source=args.tuning/name
            if target.exists() and sha(target)!=sha(source):raise ValueError('Saved tuning evidence changed')
            if not target.exists():shutil.copy2(source,target)
    # Preserve the exact executed protocol/SSC/source preparation alongside
    # aggregate metrics; later code defaults are not a protocol substitute.
    for source,name in [(directory/'protocol.json',f'{args.dataset}-st-protocol.json'),
        (directory/f'active-scope-{args.dataset}.json',f'{args.dataset}-st-scope.json'),
        (ROOT/f'rasr_research/runs/b3-source-20261124/{args.dataset}/prepared/manifest.json',
         f'{args.dataset}-st-source-preparation.json')]:
        if not source.exists():continue
        target=RECORDS/name
        if target.exists() and sha(target)!=sha(source):raise ValueError('Saved protocol changed: '+name)
        if not target.exists():shutil.copy2(source,target)
    registry=read(RECORDS/'registry.json')
    key='formal_groups' if entry['status']=='formal' else 'supporting_groups'
    group=registry.setdefault(key,[]);old=next((r for r in group if r['id']==entry['id']),None)
    if old is not None and old!=entry:raise ValueError('Registered result changed')
    if old is None:group.append(entry)
    write(RECORDS/'registry.json',registry)
    print('SAVED '+json.dumps(entry),flush=True)


def build():
    register();BACKUPS.mkdir(parents=True,exist_ok=True)
    roots=[FORMAL/'sp/seed-2024/B3',FORMAL/'us/seed-2024/B3',
        FORMAL/'sp/selection/B3',FORMAL/'us/selection/B3',
        FORMAL/'historical-checkpoint-score',ROOT/'rasr_research/runs/tuning-20261124',RECORDS]
    files=set(p for base in roots for p in base.rglob('*') if safe_artifact(p))
    files.update(p for p in FORMAL.glob('*.json') if safe_artifact(p))
    files.update(p for base in (ROOT/'SmartGen',ROOT/'rasr_research',ROOT/'tests',ROOT/'relation_research')
        for p in base.glob('*.py'))
    files.update(ROOT/name for name in ('README.md','environment.yml','baseline_snapshot.json','pytest.ini','LICENSE'))
    # Both archived source-only semantic artifacts are required for replay.
    files.update((ROOT/'relation_research/embeddings/rasr-raw-v1').glob('*_embeddings.json'))
    inventory=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    manifest={'formal_registry':read(RECORDS/'registry.json'),'files':inventory,
        'exclusions':'No raw source/target datasets, credentials, unrelated context/tof research, or downloaded language-model weights. Generated train/validation data, detector/TOF weights, all existing tuning candidates and negative results are retained.',
        'original_source_project':'/home/heyang/projects/SmartGen_GCAD (read only; required to rerun against true datasets)'}
    archive=BACKUPS/'rasr-formal-sp-us-20261124.zip'
    if archive.exists():raise ValueError('Backup already exists; verify rather than overwrite')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as bundle:
        bundle.writestr('backup-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for p in sorted(files):bundle.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:raise ValueError('ZIP integrity failure')
    write(RECORDS/'backup-receipt.json',{'archive_name':archive.name,'bytes':archive.stat().st_size,
        'sha256':sha(archive),'file_count':len(inventory),'zip_crc_verified':True,
        'raw_source_and_target_datasets_included':False,'github_uploaded':False})
    print('BACKUP '+json.dumps(read(RECORDS/'backup-receipt.json')),flush=True)


def build_next(args):
    register_next(args)
    variant='tuned' if args.tuning else 'original'
    archive=BACKUPS/f'{args.dataset}-st-b3-{variant}-20261124.zip'
    if archive.exists():raise ValueError('Existing archive retained; refusing overwrite')
    roots=[args.formal,ROOT/f'rasr_research/runs/b3-source-20261124/{args.dataset}/prepared']
    if args.tuning:roots.append(args.tuning)
    files=set(p for base in roots for p in base.rglob('*') if safe_artifact(p))
    files.update(RECORDS.glob(f'{args.dataset}-*.json'))
    files.add(RECORDS/'registry.json')
    files.add(ROOT/f'relation_research/embeddings/rasr-raw-v1/{args.dataset}_embeddings.json')
    manifest={'files':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
                       for p in sorted(files)],'raw_source_and_target_datasets_included':False}
    BACKUPS.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as bundle:
        bundle.writestr('backup-manifest.json',json.dumps(manifest,indent=2))
        for p in sorted(files):bundle.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:assert bundle.testzip() is None
    receipt={'archive_name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),
             'file_count':len(files),'zip_crc_verified':True,'github_uploaded':False}
    write(RECORDS/f'{args.dataset}-st-{variant}-backup-receipt.json',receipt)
    print('BACKUP '+json.dumps(receipt),flush=True)


def upload(args):
    """Use the existing Git credential helper in memory; never print credentials."""
    archive=BACKUPS/'rasr-formal-sp-us-20261124.zip';receipt=read(RECORDS/'backup-receipt.json')
    assert sha(archive)==receipt['sha256']
    remote=subprocess.run(['git','remote','get-url','origin'],cwd=str(ROOT),capture_output=True,text=True,check=True).stdout.strip()
    if remote!='https://github.com/heyanggg/genGcad.git':raise ValueError('Unexpected remote')
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        cwd=str(ROOT),capture_output=True,text=True,env=env,timeout=30)
    values=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    token=values.get('password')
    if credential.returncode or not token:raise RuntimeError('No noninteractive GitHub credential; no credential output saved')
    repo='heyanggg/genGcad';tag='rasr-formal-sp-us-20261124'
    def api(method,path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        request=urllib.request.Request('https://api.github.com'+path,data=data,method=method,
            headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                'User-Agent':'SmartGen-experiment-backup','Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)
    try:release=api('GET',f'/repos/{repo}/releases/tags/{tag}')
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
        release=api('POST',f'/repos/{repo}/releases',{'tag_name':tag,'target_commitish':args.commit,
            'name':'Formal RaSR groups: SP-ST and tuned US-ST (2026-11-24)',
            'body':'Two human-designated formal B3 groups, exact protocols and full generated evidence. US uses tuned detector settings and a previously exposed test; not isolated causal RaSR proof. All negative attempts retained. Raw source/target datasets and credentials excluded.',
            'draft':False,'prerelease':False})
    assets=api('GET',f'/repos/{repo}/releases/{release["id"]}/assets')
    old=next((asset for asset in assets if asset['name']==archive.name),None)
    if old is not None:
        if old['size']!=receipt['bytes']:raise ValueError('Existing asset differs; refusing replacement')
        if old.get('digest') and old['digest']!='sha256:'+receipt['sha256']:raise ValueError('Existing asset SHA mismatch')
        asset=old
    else:
        connection=http.client.HTTPSConnection('uploads.github.com',timeout=180)
        path=f'/repos/{repo}/releases/{release["id"]}/assets?name='+urllib.parse.quote(archive.name)
        connection.putrequest('POST',path)
        for name,value in {'Authorization':'Bearer '+token,'User-Agent':'SmartGen-experiment-backup',
            'Content-Type':'application/zip','Content-Length':str(receipt['bytes'])}.items():connection.putheader(name,value)
        connection.endheaders()
        with archive.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):connection.send(chunk)
        response=connection.getresponse();raw=response.read()
        if response.status!=201:raise RuntimeError(f'Asset upload HTTP {response.status}; no automatic retry')
        asset=json.loads(raw);connection.close()
    assert asset['size']==receipt['bytes']
    if asset.get('digest'):assert asset['digest']=='sha256:'+receipt['sha256']
    receipt.update(github_uploaded=True,release_url=release['html_url'],asset_url=asset['browser_download_url'],
        github_asset_size=asset['size'],github_digest=asset.get('digest'),source_commit=args.commit)
    write(RECORDS/'backup-receipt.json',receipt)
    print('GITHUB BACKUP VERIFIED '+json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['register','register-next','build','build-next','upload','git-export'],required=True)
    parser.add_argument('--commit',help='Already-pushed immutable commit for release tag')
    parser.add_argument('--batch',type=int,default=0)
    parser.add_argument('--utf8',action='store_true',help='Inline text tree export for the connected GitHub API')
    parser.add_argument('--dataset',choices=['fr'],default='fr')
    parser.add_argument('--formal',type=Path,default=ROOT/'rasr_research/runs/formal-fr-st-20261124')
    parser.add_argument('--tuning',type=Path)
    parser.add_argument('--since',help='Export only paths changed since this local commit')
    args=parser.parse_args()
    args.formal=args.formal.resolve()
    if args.tuning:args.tuning=args.tuning.resolve()
    if args.action=='upload':
        if not args.commit or not re.fullmatch(r'[0-9a-f]{40}',args.commit):parser.error('--commit must be a pushed commit SHA')
        upload(args)
    elif args.action=='register-next':register_next(args)
    elif args.action=='build-next':build_next(args)
    elif args.action=='git-export':
        command=['git','diff','--name-only','-z',args.since,'HEAD'] if args.since else ['git','ls-files','-z']
        paths=subprocess.run(command,cwd=str(ROOT),capture_output=True,check=True).stdout.decode().split('\0')
        paths=[p for p in paths if p]
        # Read the committed blobs, not potentially-changing working-tree files.
        records=[]
        for path in paths[args.batch*4:(args.batch+1)*4]:
            blob=subprocess.run(['git','show','HEAD:'+path],cwd=str(ROOT),capture_output=True,check=True).stdout
            records.append({'path':path,'content':blob.decode('utf-8') if args.utf8 else base64.b64encode(blob).decode('ascii'),
                'git_sha':hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()})
        print(json.dumps({'total':len(paths),'batch':args.batch,'files':records}))
    else:{'register':register,'build':build}[args.action]()


if __name__=='__main__':main()
