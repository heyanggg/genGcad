"""Reevaluate SAVED legacy generation when detector splits were not archived.

No generation, no TOF rerun, no reconstruction of an unverifiable old split.
Keep the original checkpoint/metrics immutable and explicitly separate.
"""
import argparse
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import shutil
import torch

from formal_experiment import (Anomaly_detection, ROOT, atomic_write_json, cwd,
    load, read, sha, target_copies)


def reevaluate(args):
    archive, = (args.source/'experiment_archive/formal').glob(f'{args.dataset}_single_to_multiple_*')
    original = read(archive/'artifacts/anomaly_detection/metrics.json')
    config = read(archive/'manifest.json')['config']
    final, = (archive/'artifacts/tof').glob('*_seq_filter_true.pkl')
    output = args.output/'baseline-reevaluation'/args.dataset
    receipt = output/'result.json'
    if receipt.exists():
        saved = read(receipt)
        assert saved['input_sha256'] == sha(final)
        for path, digest in saved['files_sha256'].items(): assert sha(path) == digest
        return saved
    assert not (archive/'artifacts/anomaly_detection/train.pkl').exists()
    assert not (archive/'artifacts/anomaly_detection/validation.pkl').exists()
    runtime = output/'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    target_copies(args, args.dataset, runtime, 'multiple')
    label = 'legacy-existing-data-reevaluation'
    destination = runtime/f'filter_data/{args.dataset}/multiple/{args.dataset}_multiple_generation_SPPC_th={config["threshold"]}_{label}_seq_filter_true.pkl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final, destination)
    with (output/'evaluation.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log), cwd(runtime):
        detection = Anomaly_detection(args.dataset, 'multiple', config['threshold'], 'SPPC', label,
            config['validation_percentile'], seed=2024)
    detector = runtime/Path(detection['metrics_path']).parent
    saved = {'mode': 'Existing historical post-TOF generation; NEW detector evaluation, NOT recovery of historical splits',
        'dataset': args.dataset, 'new_codex_calls': 0, 'TOF_rerun': False,
        'input': str(final), 'input_sha256': sha(final), 'input_count': len(load(final)),
        'original_record': original, 'historical_split_recovered': False,
        'reason': 'Legacy archives and original anomaly_runs contain no saved detector train/validation files; recorded train=N, validation=N, generated=N is inconsistent. Original split and historical training protocol are unverifiable.',
        'protocol': {'epochs': 15, 'learning_rate': .001, 'train_ratio': .8, 'seed': 2024,
            'validation_percentile': config['validation_percentile'], 'core_algorithm_changed': False},
        'detector_dir': str(detector), 'detection': detection,
        'files_sha256': {str(p): sha(p) for p in [final, archive/'artifacts/anomaly_detection/metrics.json',
            archive/'artifacts/anomaly_detection/detector.pth', *sorted(detector.iterdir())]},
        'selection_uses_target_test': False, 'original_source_modified': False}
    atomic_write_json(receipt, saved)
    print(json.dumps({'mode': saved['mode'], 'detection': detection}, indent=2), flush=True)
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sp'], required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/heyang/projects/SmartGen_GCAD'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output = args.output.resolve()
    torch.set_num_threads(2)
    reevaluate(args)
