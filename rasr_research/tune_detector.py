"""Offline, generated-validation-only tuning of the EXISTING detector.

No Codex, target environment fitting, new network or modified training loss.
Synthetic validation negatives are explicitly proxies, not real anomaly labels.
Target test files are accessed only by a separately invoked locked confirmation.
"""
import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'SmartGen'))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
import dictionary
from baseline1 import setup_seed
from models1 import TransformerAutoencoder
from experiment import atomic_pickle_dump,atomic_write_json


def load(path):
    with Path(path).open('rb') as handle:return pickle.load(handle)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tokens(row,feature_index=3):return tuple(row[feature_index:40:4])


def feature_index(environment):
    return {'spring':3,'night':1,'multiple':2}[environment]


def vocabulary_size(ds):return {'sp':235,'us':269,'fr':223}[ds]


def coverage_split(rows,seed,ratio=.2,feature_index=3):
    """Exact duplicates never cross; retain support for every validation action.

    Singleton-action groups remain training-only. This uses normal rows only,
    and does not assume anything about seasons or target anomaly definitions.
    """
    groups=defaultdict(list)
    for row in rows:groups[tuple(row)].append(list(row))
    keys=sorted(groups)
    random.Random(seed).shuffle(keys)
    supports=Counter(a for key in keys for a in set(tokens(key,feature_index)))
    target=max(1,round(len(rows)*ratio));validation=[];held=set()
    for key in keys:
        if len(validation)>=target:break
        acts=set(tokens(key,feature_index))
        if len(validation)+len(groups[key])>target:continue
        if all(supports[a]>1 for a in acts):
            held.add(key);validation.extend(groups[key]);supports.subtract(acts)
    training=[r for key in keys if key not in held for r in groups[key]]
    if not training or not validation:raise ValueError('Insufficient independent normal groups for covered validation')
    assert not {tuple(r) for r in training}&{tuple(r) for r in validation}
    assert {a for r in validation for a in tokens(r,feature_index)}<={a for r in training for a in tokens(r,feature_index)}
    return training,validation


def arrays(rows,vocab,feature_index=3):
    out=np.full((len(rows),10),vocab-1,dtype=np.int64)
    for i,row in enumerate(rows):
        values=tokens(row,feature_index)
        if not values:raise ValueError('empty behavior row')
        out[i,:len(values)]=values
    return out


def proxy_rows(validation,normal_pool,legal_actions,seed,feature_index=3):
    """Uniform unseen-action corruption, no target labels or named attack types."""
    seen={a for r in normal_pool for a in tokens(r,feature_index)}
    candidates=sorted(set(legal_actions)-seen)
    if not candidates:raise ValueError('No unseen legal actions for proxy validation')
    rng=random.Random(seed);output=[]
    for row in validation:
        for _ in range(3):
            changed=list(row);position=rng.randrange(min(10,len(row)//4))
            changed[position*4+feature_index]=rng.choice(candidates)
            output.append(changed)
    return output


def normal_views(rows):
    """Source-TSS-like short normal views, derived ONLY after group holdout."""
    result=[]
    for row in rows:
        variants={tuple(row)}
        for length in range(1,min(10,len(row)//4)+1):variants.add(tuple(row[:length*4]))
        for offset in range(0,min(40,len(row)),4):variants.add(tuple(row[offset:offset+4]))
        result.extend(list(r) for r in sorted(variants))
    return result


def stress(args):
    selection=read(args.output/'selection.json')
    ds=args.dataset;vocab=vocabulary_size(ds)
    source_root='b3-source-20261124' if ds=='fr' else 'pilot-20261124'
    spec=read(args.formal/'protocol.json')['datasets'][ds]
    source_rows=load(Path(spec.get('prepared_source',ROOT/f'rasr_research/runs/{source_root}/{ds}/prepared'))/'I.pkl')
    field=selection.get('feature_index',3)
    result={'source_TSS_lengths':dict(Counter(len(r['row'])//4 for r in source_rows)),
        'selection_sha256':sha(args.output/'selection.json'),'test_files_accessed':False,'arms':{}}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for arm in ('historical','B3'):
        folder=args.output/arm;cal=read(folder/'calibration.json')
        validation=load(folder/'validation.pkl');views=normal_views(validation)
        model=model_for(vocab,device);model.load_state_dict(torch.load(folder/'detector.pth',map_location=device))
        values=score(model,views,vocab,device,field)[:,['mean','max','top2'].index(selection['parameters']['score'])]
        result['arms'][arm]={'original_validation_count':len(validation),'normal_views':len(views),
            'old_threshold':cal['threshold'],'normal_view_fpr':float(np.mean(values>=cal['threshold'])),
            'view_p95':float(np.percentile(values,95))}
        del model;torch.cuda.empty_cache()
    atomic_write_json(args.output/'normal_view_stress.json',result)
    print(json.dumps(result,indent=2))


def score(model,rows,vocab,device,feature_index=3):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(rows),64):
            src=torch.as_tensor(arrays(rows[start:start+64],vocab,feature_index),device=device)
            valid=src.ne(vocab-1)
            logits=model(src,src_key_padding_mask=~valid)
            losses=F.cross_entropy(logits.transpose(1,2),src,reduction='none')
            means=(losses*valid).sum(1)/valid.sum(1)
            maxima=losses.masked_fill(~valid,-torch.inf).max(1).values
            top=losses.masked_fill(~valid,-torch.inf).topk(2,dim=1).values
            top2=torch.where(torch.isfinite(top),top,torch.zeros_like(top)).sum(1)/valid.sum(1).clamp(max=2)
            values.append(torch.stack([means,maxima,top2],1).cpu().numpy())
    return np.concatenate(values)


def model_for(vocab,device):
    return TransformerAutoencoder(vocab,d_model=512,nhead=8,num_encoder_layers=2,num_decoder_layers=2).to(device)


def train_epochs(model,rows,vocab,device,optimizer,start,end,seed,feature_index=3):
    data=arrays(rows,vocab,feature_index)
    for epoch in range(start,end):
        model.train();losses=[]
        # Original loader order retained; only explicit configuration changes.
        for begin in range(0,len(data),32):
            src=torch.as_tensor(data[begin:begin+32],device=device)
            valid=src.ne(vocab-1)
            logits=model(src,src_key_padding_mask=~valid)
            element=F.cross_entropy(logits.transpose(1,2),src,reduction='none')
            loss=(element*valid).sum()/valid.sum()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss))
        yield epoch+1,float(np.mean(losses))


def inputs(args):
    spec=read(args.formal/'protocol.json')['datasets'][args.dataset]
    archive=(Path(spec['baseline_reevaluation']['detector_dir']) if 'baseline_reevaluation' in spec
             else Path(spec['archive'])/'artifacts/anomaly_detection')
    b3=read(args.formal/f'{args.dataset}/seed-2024/B3/result.json')
    folder=args.formal/f'{args.dataset}/seed-2024/B3/runtime'/Path(b3['detection']['metrics_path']).parent
    files={'historical':[archive/'train.pkl',archive/'validation.pkl'],
           'B3':[folder/'train.pkl',folder/'validation.pkl']}
    rows={arm:load(paths[0])+load(paths[1]) for arm,paths in files.items()}
    return spec,rows,{str(p):sha(p) for paths in files.values() for p in paths}


def fit(args):
    output=args.output
    if (output/'selection.json').exists():raise ValueError('Selection already locked; do not silently retune')
    spec,pools,fingerprints=inputs(args)
    ds=args.dataset;vocab=vocabulary_size(ds)
    environment=spec['reference_config'].get('target_environment','spring');field=feature_index(environment)
    legal=([value for name,value in getattr(dictionary,f'{ds}_actions').items() if not name.startswith('None:')]
           if field==3 else list(dictionary.hour_dict.values()) if field==1 else list(getattr(dictionary,f'{ds}_devices_dict').values()))
    grid={'learning_rate':[.0003,.001],'weight_decay':[0.,.0001],
          'epochs':[5,10,15,25],'score':['mean','max','top2'],'percentile':[90.,95.,97.5]}
    protocol={'dataset':ds,'input_sha256':fingerprints,'grid':grid,'cv_seeds':[2024,2025,2026],
        'environment':environment,'feature_index':field,
        'architecture':'unchanged 512D 8-head 2 encoder/2 decoder existing TransformerAutoencoder',
        'loss':'unchanged masked normal reconstruction cross-entropy','batch_size':32,
        'selection':'mean proxy F1 over BOTH arms and three generated-normal holdouts; ties normal mean loss then original-parameter proximity',
        'proxy':'uniform one-action substitution from legal dictionary actions absent in the normal generated corpus; NOT real labeled anomalies',
        'test_files_accessed_for_selection':False,'test_previously_exposed_in_earlier_experiments':True,
        'new_model_or_generator':False,'new_codex_calls':0,'holdout_note':'exact-duplicate groups disjoint; singleton-action groups kept in training',
        'calibration_views':args.calibration_views,
        'view_independence_warning':'Multiple normal views share held-out parent rows; view count is not independent sample size'}
    if field!=3:
        protocol['proxy']='Uniform substitution of the existing detector input field with a legal dictionary value unseen in each generated-normal pool. No target test rows or attack labels accessed. If either pool has no unseen legal values, stop without inventing proxies or changing the search.'
        protocol['holdout_note']='Exact full-row duplicate groups disjoint; singleton input-field supports kept in training'
    atomic_write_json(output/'protocol.json',protocol)
    observations=defaultdict(list);curves=[];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for lr in grid['learning_rate']:
        for decay in grid['weight_decay']:
            for seed in protocol['cv_seeds']:
                for arm,rows in pools.items():
                    training,validation=coverage_split(rows,seed,feature_index=field)
                    calibration=normal_views(validation) if args.calibration_views else validation
                    negatives=proxy_rows(calibration,rows,legal,seed,field)
                    setup_seed(seed);model=model_for(vocab,device)
                    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
                    for epoch,loss in train_epochs(model,training,vocab,device,optimizer,0,max(grid['epochs']),seed,field):
                        curves.append({'arm':arm,'seed':seed,'lr':lr,'weight_decay':decay,'epoch':epoch,'train_loss':loss})
                        if epoch not in grid['epochs']:continue
                        normal_scores=score(model,calibration,vocab,device,field);proxy_scores=score(model,negatives,vocab,device,field)
                        for index,mode in enumerate(grid['score']):
                            for percentile in grid['percentile']:
                                cutoff=float(np.percentile(normal_scores[:,index],percentile))
                                labels=np.r_[np.zeros(len(calibration)),np.ones(len(negatives))]
                                predictions=np.r_[normal_scores[:,index],proxy_scores[:,index]]>=cutoff
                                key=(lr,decay,epoch,mode,percentile)
                                observations[key].append({'arm':arm,'seed':seed,'normal_count':len(calibration),
                                    'proxy_count':len(negatives),'proxy_f1':float(f1_score(labels,predictions)),
                                    'normal_mean_loss':float(normal_scores[:,0].mean()),'threshold':cutoff,
                                    'normal_fpr':float(np.mean(normal_scores[:,index]>=cutoff))})
                    print(f'CV {arm} seed={seed} lr={lr} wd={decay} complete; no target test access',flush=True)
                    del optimizer,model;torch.cuda.empty_cache()
                    atomic_write_json(output/'training_curves.json',curves)
    candidates=[]
    for key,values in observations.items():
        lr,decay,epoch,mode,pct=key
        candidates.append({'learning_rate':lr,'weight_decay':decay,'epochs':epoch,'score':mode,'percentile':pct,
            'mean_proxy_f1':float(np.mean([r['proxy_f1'] for r in values])),
            'std_proxy_f1':float(np.std([r['proxy_f1'] for r in values])),
            'normal_mean_loss':float(np.mean([r['normal_mean_loss'] for r in values])),
            'folds':values})
    candidates.sort(key=lambda c:(-c['mean_proxy_f1'],c['normal_mean_loss'],
        abs(c['percentile']-95),c['score']!='mean',abs(c['learning_rate']-.001),c['epochs']))
    atomic_write_json(output/'all_candidates.json',candidates)
    selected={k:candidates[0][k] for k in grid}
    selection={'parameters':selected,'input_sha256':fingerprints,'selection_metric':candidates[0]['mean_proxy_f1'],
        'test_not_used_to_select':True,'new_codex_calls':0,'candidate_count':len(candidates),'calibration_views':args.calibration_views}
    if field!=3:selection.update(environment=environment,feature_index=field)
    atomic_write_json(output/'selection.json',selection)  # lock BEFORE final refits or test scoring
    for arm,rows in pools.items():
        directory=output/arm;directory.mkdir(parents=True,exist_ok=True)
        training,validation=coverage_split(rows,2024,feature_index=field)
        atomic_pickle_dump(directory/'train.pkl',training);atomic_pickle_dump(directory/'validation.pkl',validation)
        setup_seed(2024);model=model_for(vocab,device)
        optimizer=torch.optim.Adam(model.parameters(),lr=selected['learning_rate'],weight_decay=selected['weight_decay'])
        for epoch,loss in train_epochs(model,training,vocab,device,optimizer,0,selected['epochs'],2024,field):pass
        torch.save(model.state_dict(),directory/'detector.pth')
        calibration=normal_views(validation) if args.calibration_views else validation
        validation_scores=score(model,calibration,vocab,device,field)
        index=grid['score'].index(selected['score'])
        threshold=float(np.percentile(validation_scores[:,index],selected['percentile']))
        np.savez_compressed(directory/'validation_scores.npz',scores=validation_scores)
        atomic_write_json(directory/'calibration.json',{'threshold':threshold,'parameters':selected,
            'train_count':len(training),'validation_count':len(validation),'calibration_view_count':len(calibration),
            'selection_sha256':sha(output/'selection.json')})
        del optimizer,model;torch.cuda.empty_cache()
    print('LOCKED '+json.dumps(selection),flush=True)


def confirm(args):
    if (args.output/'confirmation.json').exists():raise ValueError('Confirmation already exists; no adaptive rerun')
    selection=read(args.output/'selection.json')
    for path,digest in selection['input_sha256'].items():assert sha(path)==digest
    ds=args.dataset;vocab=vocabulary_size(ds)
    source=Path('/home/heyang/projects/SmartGen_GCAD/SmartGen')
    # Only this post-lock action is allowed to read the target test files.
    environment=selection.get('environment','spring');field=feature_index(environment)
    attack_suffix={'spring':'heater','night':'time','multiple':'tv'}[environment]
    normal=load(source/f'IoT_data/{ds}/{environment}/split_test.pkl')
    labeled=load(source/f'attack/{ds}/labeled_{ds}_{environment}_attack_{attack_suffix}.pkl')
    rows=normal+[r[0] for r in labeled];labels=np.r_[np.zeros(len(normal)),[r[1] for r in labeled]]
    report={'dataset':ds,'selection_sha256':sha(args.output/'selection.json'),'selection':selection,
        'limitations':'Previously exposed test; exploratory confirmation, NOT a fresh confirmatory holdout. Proxy validation negatives are not real anomaly labels.',
        'new_codex_calls':0,'arms':{}}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for arm in ('historical','B3'):
        directory=args.output/arm;calibration=read(directory/'calibration.json')
        assert calibration['selection_sha256']==report['selection_sha256']
        model=model_for(vocab,device);model.load_state_dict(torch.load(directory/'detector.pth',map_location=device))
        scores=score(model,rows,vocab,device,field)[:,['mean','max','top2'].index(selection['parameters']['score'])]
        pred=scores>=calibration['threshold'];positive=labels==1;negative=labels==0
        metrics={'TP':int(np.sum(pred&positive)),'TN':int(np.sum(~pred&negative)),
            'FP':int(np.sum(pred&negative)),'FN':int(np.sum(~pred&positive)),
            'f1_score':float(f1_score(labels,pred)),'precision':float(precision_score(labels,pred)),
            'recall':float(recall_score(labels,pred)),'FPR':float(np.mean(pred[negative])),
            'FNR':float(np.mean(~pred[positive])),'auroc':float(roc_auc_score(labels,scores)),
            'auprc':float(average_precision_score(labels,scores))}
        np.savez_compressed(directory/'continuous_scores.npz',labels=labels,scores=scores,predictions=pred)
        report['arms'][arm]={'metrics':metrics,'calibration':calibration}
        print(f'CONFIRM {arm}: '+json.dumps(metrics),flush=True)
        del model;torch.cuda.empty_cache()
    atomic_write_json(args.output/'confirmation.json',report)


def report(args):
    """Summarize preserved evaluations; never select or retrain from test results."""
    confirmed=read(args.output/'confirmation.json')
    assert confirmed['selection_sha256']==sha(args.output/'selection.json')
    suffix='-'+args.dataset if args.dataset!='sp' else ''
    comparison=read(args.formal/f'single-b3-comparison{suffix}.json')
    old_b3=comparison['B3_metrics']
    old_baseline=comparison['comparison_baseline_metrics']
    tuned_b3=confirmed['arms']['B3']['metrics']
    tuned_baseline=confirmed['arms']['historical']['metrics']
    result={'dataset':args.dataset,'new_codex_calls':0,
        'protocol':'Offline detector tuning, NOT an unchanged-formal-protocol RaSR comparison',
        'parameters':confirmed['selection']['parameters'],
        'metrics':{'original_historical_B0':old_baseline,'original_B3':old_b3,
            'common_tuned_historical_data_B0':tuned_baseline,'common_tuned_B3':tuned_b3},
        'f1_delta_percentage_points':{
            'B3_tuning_vs_original_B3':100*(tuned_b3['f1_score']-old_b3['f1_score']),
            'tuned_B3_vs_original_historical_B0':100*(tuned_b3['f1_score']-old_baseline['f1_score']),
            'tuned_B3_vs_common_tuned_B0':100*(tuned_b3['f1_score']-tuned_baseline['f1_score'])},
        'interpretation':'Detector recovery alone is not evidence that RaSR caused the recovery. The common-tuned B3/B0 comparison is single-seed and unreplicated; retain its actual difference and both protocols.',
        'limitations':[
            confirmed['limitations'],
            'One final evaluation seed; three holdout seeds are validation, not three independent final experiments.',
            'Short normal views share parent sequences, not independent calibration observations.',
            'Covered validation keeps singleton-action groups training-only; proxy negatives represent unseen-action corruption, not all real attacks.',
            'Generated sample counts and actual SSC prompt exposure must be checked in the scenario comparison; equality or difference is scenario-specific.',
            'Threshold, score aggregation, grouped split and training settings changed: keep separate from original formal results.',
            'F1 gains must be considered alongside false-positive and false-negative rates, not alone.'],
        'evidence':{'confirmation_sha256':sha(args.output/'confirmation.json'),
            'selection_sha256':confirmed['selection_sha256'],
            'input_sha256':confirmed['selection']['input_sha256'],
            'model_sha256':{arm:sha(args.output/arm/'detector.pth') for arm in ('historical','B3')}}}
    if confirmed['selection'].get('feature_index',3)!=3:
        result['limitations'][3]='Covered validation retains singleton input-field supports in training. Proxies corrupt only the original detector input field; they are not real anomaly labels.'
    if comparison.get('historical_protocol_caveat'):
        result['historical_protocol_caveat']=comparison['historical_protocol_caveat']
        result['metrics']['matched_protocol_existing_data_B0']=comparison['matched_protocol_baseline']['metrics']
        result['limitations'].append(comparison['historical_protocol_caveat'])
    if (comparison.get('gcad_comparison_state') or {}).get('guidance_enabled') is False:
        result['gcad_comparison_state']=comparison['gcad_comparison_state']
        result['limitations'].append('Historical GCAD-auto disabled guidance; this is not an active-GCAD three-module comparison.')
        if result['gcad_comparison_state'].get('mode_field_missing_in_legacy_config'):
            result['limitations'][-1]='Historical artifact confirms disabled GCAD; no mode flag was recorded. Not an active-GCAD three-module comparison.'
    atomic_write_json(args.output/'summary.json',result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['fit','confirm','stress','report'],required=True)
    parser.add_argument('--dataset',choices=['sp','us','fr'],default='us')
    parser.add_argument('--formal',type=Path,default=ROOT/'rasr_research/runs/formal-20261124')
    parser.add_argument('--output',type=Path,default=ROOT/'rasr_research/runs/tuning-20261124/covered-cv')
    parser.add_argument('--calibration-views',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    {'fit':fit,'confirm':confirm,'stress':stress,'report':report}[args.action](args)


if __name__=='__main__':main()
