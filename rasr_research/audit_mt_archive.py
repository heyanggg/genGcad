"""Read-only inventory of legacy single-to-multiple baseline evidence."""
import json
import pickle
import argparse
from pathlib import Path

SOURCE = Path('/home/heyang/projects/SmartGen_GCAD')


def normal_split_audit(formal,dataset='sp'):
    """Inspect only generated normals and calibration; no target test access."""
    import numpy as np
    receipt=json.loads((formal/f'baseline-reevaluation/{dataset}/result.json').read_text())
    folder=Path(receipt['detector_dir'])
    def rows(path):
        with path.open('rb') as handle:return pickle.load(handle)
    training,validation=rows(folder/'train.pkl'),rows(folder/'validation.pkl')
    seen={int(v) for r in training for v in r[2:40:4]}
    scores=np.load(folder/'validation_scores.npz')['scores']
    report={'target_test_files_accessed':False,'training_count':len(training),
        'validation_count':len(validation),'training_device_values':sorted(seen),
        'validation_devices_not_in_training':sorted({int(v) for r in validation for v in r[2:40:4]}-seen),
        'validation_loss_quantiles':{str(p):float(np.percentile(scores,p)) for p in (0,50,90,95,99,100)},
        'validation_rows_with_unseen_devices':[{'index':i,'device_values':list(r[2:40:4]),'loss':float(scores[i])}
            for i,r in enumerate(validation) if set(r[2:40:4])-seen],
        'train_validation_exact_row_overlap':len({tuple(r) for r in training}&{tuple(r) for r in validation}),
        'interpretation':'Normal-only coverage and calibration diagnostic; not a test-guided choice of percentile, model or generation settings.'}
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal',type=Path)
    parser.add_argument('--dataset',choices=['sp','us','fr'])
    args=parser.parse_args()
    if args.formal:
        normal_split_audit(args.formal,args.dataset or 'sp')
        return
    for ds in ((args.dataset,) if args.dataset else ('sp', 'us', 'fr')):
        archive, = (SOURCE/'experiment_archive/formal').glob(f'{ds}_single_to_multiple_*')
        manifest = json.loads((archive/'manifest.json').read_text())
        metrics = json.loads((archive/'artifacts/anomaly_detection/metrics.json').read_text())
        print(json.dumps({'dataset': ds, 'manifest_keys': list(manifest),
            'metrics': metrics, 'provenance': {k:v for k,v in manifest.items()
                if k not in ('config','files','generation','provenance')},
            'provenance_keys': list(manifest.get('provenance', {})),
            'anomaly_files': [f for f in manifest.get('files', [])
                if 'anomaly_detection' in str(f.get('archive', f.get('destination','')))]}, indent=2))
        for path in sorted((archive/'artifacts/tof').glob('*.pkl')):
            if '_out_' in path.name: continue
            with path.open('rb') as handle: rows = pickle.load(handle)
            print(path.name, len(rows))
        print('original_detector_files', [str(p.relative_to(SOURCE)) for p in
            (SOURCE/'SmartGen/anomaly_runs').glob(f'{ds}_multiple*/*')])


if __name__ == '__main__':
    main()
