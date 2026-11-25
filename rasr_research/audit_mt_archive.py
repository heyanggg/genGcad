"""Read-only inventory of legacy single-to-multiple baseline evidence."""
import json
import pickle
from pathlib import Path

SOURCE = Path('/home/heyang/projects/SmartGen_GCAD')


def main():
    for ds in ('sp', 'us', 'fr'):
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
