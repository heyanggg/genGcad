"""Read-only error/data audit. No fitting, generation or parameter selection."""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SmartGen'))
import dictionary
from experiment import atomic_write_json


def load(path):
    with Path(path).open('rb') as handle:return pickle.load(handle)


def read(path):return json.loads(Path(path).read_text())


def actions(row):return list(row[3:40:4])


def corpus(rows,names):
    counts=Counter(a for row in rows for a in actions(row))
    return {'sequences':len(rows),'unique_sequences':len({tuple(r) for r in rows}),
        'events':sum(counts.values()),'unique_actions':len(counts),
        'lengths':dict(Counter(len(r)//4 for r in rows)),
        'actions':{names.get(a,str(a)):n for a,n in counts.most_common()},
        'heater_sequences':sum(any(names.get(a,'').startswith('Heater:') for a in actions(r)) for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['sp','us','fr'],default='us')
    parser.add_argument('--normal-only',action='store_true',help='Do not read even cached target test scores')
    parser.add_argument('--root',type=Path,default=ROOT/'rasr_research/runs/formal-20261124')
    parser.add_argument('--output',type=Path,default=ROOT/'rasr_research/runs/tuning-20261124/diagnosis.json')
    args=parser.parse_args();ds=args.dataset
    protocol=read(args.root/'protocol.json');spec=protocol['datasets'][ds]
    names={v:k for k,v in getattr(dictionary,f'{ds}_actions').items()}
    archive=Path(spec['archive'])
    result={'dataset':ds,'mode':'descriptive diagnostics only, not a parameter selector',
        'test_already_exposed':True,'cached_target_scores_read':not args.normal_only,'arms':{}}
    for arm in ('historical','B0','B3'):
        if arm=='historical':
            folder=archive/'artifacts/anomaly_detection';metric=spec['reference_metrics']
            scores=args.root/f'historical-checkpoint-score/{ds}/continuous_scores.npz'
            vld_scores=None
        else:
            run=args.root/f'{ds}/seed-2024/{arm}'
            if not (run/'result.json').exists():continue
            record=read(run/'result.json');metric=record['detection']
            folder=run/'runtime'/Path(metric['metrics_path']).parent
            scores=run/'runtime'/metric['continuous_scores_path']
            vld_scores=run/'runtime'/metric['validation_scores_path']
        training=load(folder/'train.pkl');validation=load(folder/'validation.pkl')
        seen=Counter(a for r in training for a in actions(r))
        item={'training':corpus(training,names),'validation':corpus(validation,names),
            'threshold':metric['detection_threshold'],'metrics':metric['metrics'],
            'duplicate_train_validation_rows':len({tuple(r) for r in training}&{tuple(r) for r in validation}),
            'validation_actions_unseen_in_train':dict(Counter(names.get(a,str(a)) for r in validation for a in actions(r) if not seen[a]))}
        if vld_scores:
            values=np.load(vld_scores)['scores']
            item['validation_loss_quantiles']={str(q):float(np.percentile(values,q)) for q in (0,50,75,90,95,100)}
            item['validation_tail']=[{'loss':float(values[i]),'actions':[names.get(a,str(a)) for a in actions(validation[i])],
                'train_action_counts':[seen[a] for a in actions(validation[i])]} for i in np.argsort(values)[-5:][::-1]]
        if not args.normal_only:
            stored=np.load(scores);labels=stored['labels'];losses=stored['scores']
            item['test_descriptive_score_quantiles']={str(label):{str(q):float(np.percentile(losses[labels==label],q))
                for q in (0,10,25,50,75,90,95,100)} for label in (0,1)}
        result['arms'][arm]=item
    atomic_write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
