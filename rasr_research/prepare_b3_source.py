"""Freeze source-only RaSR inputs for the next B3, without running ablations."""
import argparse
import json
import pickle
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SmartGen'))
import numpy as np
from main import ACTION_DICTIONARIES, DEVICE_DICTIONARIES
from rasr import RaSRConfig,prepare_source,split_parent_groups,build_relations,augment
from experiment import atomic_pickle_dump,atomic_write_json
from formal_experiment import sha,load


def prepare(source,dataset,output):
    raw=source/f'SmartGen/IoT_data/{dataset}/winter/trn.pkl'
    tss=raw.parent/'split_trn.pkl'
    embeddings=ROOT/f'relation_research/embeddings/rasr-raw-v1/{dataset}_embeddings.json'
    ready=output/'manifest.json'
    if ready.exists():
        saved=json.loads(ready.read_text())
        for path,digest in saved['input_sha256'].items():assert sha(path)==digest
        for name,digest in saved['output_sha256'].items():assert sha(output/name)==digest
        return saved
    actions,devices=ACTION_DICTIONARIES[dataset],DEVICE_DICTIONARIES[dataset]
    originals,names,action_devices,candidates=prepare_source(load(raw),dataset,actions,devices)
    assert [r['row'] for r in originals]==load(tss)
    config=RaSRConfig();training,validation=split_parent_groups(originals,config)
    assert not {r['parent_group'] for r in training}&{r['parent_group'] for r in validation}
    artifact=json.loads(embeddings.read_text())
    assert artifact['model']['text_profile']=='raw-action'
    vectors=dict(zip(artifact['action_names'],artifact['vectors']))
    relations=build_relations(training,[vectors[name] for name in names],config)
    variants,skips=augment(training,relations,candidates,[actions[n] for n in names],action_devices,config)
    assert {r['parent_id'] for r in variants}<={r['id'] for r in training}
    output.mkdir(parents=True,exist_ok=True)
    atomic_pickle_dump(output/'I.pkl',originals);atomic_pickle_dump(output/'B3_J.pkl',variants)
    np.savez_compressed(output/'relations.npz',**relations)
    atomic_write_json(output/'split.json',{'train_ids':[r['id'] for r in training],
        'validation_ids':[r['id'] for r in validation],
        'train_parent_groups':sorted({r['parent_group'] for r in training}),
        'validation_parent_groups':sorted({r['parent_group'] for r in validation})})
    saved={'dataset':dataset,'config':asdict(config),'source_count':len(originals),
        'train_count':len(training),'validation_count':len(validation),'B3_J_count':len(variants),
        'operation_counts':dict(Counter(r['edit']['operation'] for r in variants)),'skipped':skips,
        'input_sha256':{str(p):sha(p) for p in (raw,tss,embeddings,ROOT/'SmartGen/rasr.py')},
        'output_sha256':{name:sha(output/name) for name in ('I.pkl','B3_J.pkl','relations.npz','split.json')},
        'adaptation':'Only B3 is requested: keep its valid existing-algorithm variants, no downsampling to unrun B1/B2 arms. No RaSR algorithm or parameters changed.',
        'test_or_target_data_used':False,'new_codex_calls':0}
    atomic_write_json(ready,saved);return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/home/heyang/projects/SmartGen_GCAD'))
    parser.add_argument('--dataset',choices=['fr'],default='fr')
    parser.add_argument('--output',type=Path,default=ROOT/'rasr_research/runs/b3-source-20261124/fr/prepared')
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.dataset,args.output),indent=2))


if __name__=='__main__':main()
