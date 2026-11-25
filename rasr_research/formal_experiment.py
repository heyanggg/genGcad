"""Frozen-algorithm formal B3 compared with the original historical baseline.

Archived *actual prompts*, not potentially stale SSC directory contents, define
the historical baseline and the day/category/example budgets. The archived SSC
backbone is a common source-trained component, as in the archived US protocol.
The earlier multi-arm/repetition schedule is historical, not an active entry point.
Only the dataset explicitly listed in generation-policy.json is authorized for
a single B3 seed-2024 run; historical B0 requires no generation calls.
No target test data are consulted to select examples or adjust parameters.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SmartGen"))
import numpy as np
import torch
from sklearn.preprocessing import normalize
from baseline1 import Anomaly_detection, evaluate as evaluate_detector
from codex_backend import CodexClient
import dictionary
from experiment import atomic_pickle_dump, atomic_write_json, atomic_write_text
from extract import Extract
from security_check import security_check
from sppc import TransformerAutoencoder, setup_seed
from transnumber import Transnum

MARKER = "The user's compressed original sequences of behavior: "
END = ". User's behavior habits: "
SEEDS = (2024, 2025, 2026)
ARMS = ("B0", "B1", "B3")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def prepared_path(ds, source_env='winter'):
    if source_env!='winter':
        return ROOT/f'rasr_research/runs/b3-source-20261125/{ds}-{source_env}/prepared'
    root='b3-source-20261124' if ds=='fr' else 'pilot-20261124'
    return ROOT/f'rasr_research/runs/{root}/{ds}/prepared'


def vocabulary_size(ds):
    return {'sp':235,'us':269,'fr':223}[ds]


def frozen_files(datasets=('sp','us'),source_env='winter'):
    return {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "SmartGen/rasr.py",
        *[ROOT/f'relation_research/embeddings/rasr-raw-v1/{ds}_embeddings.json' for ds in sorted(datasets)],
        *[prepared_path(ds,source_env)/name for ds in datasets
          for name in (('relations.npz','B3_J.pkl','I.pkl','split.json') if ds=='fr' or source_env!='winter' else
                       ('relations.npz','B1_J.pkl','B3_J.pkl','I.pkl'))]]}


def target_environment(protocol,ds):
    return protocol['datasets'][ds].get('reference_config',{}).get('target_environment','spring')


def test_relatives(ds,environment):
    attack={'spring':'heater','night':'time','multiple':'tv'}[environment]
    return (f'IoT_data/{ds}/{environment}/split_test.pkl',
            f'attack/{ds}/labeled_{ds}_{environment}_attack_{attack}.pkl')


def scenario(protocol,ds):
    spec=protocol['datasets'][ds]
    return ds.upper()+'-'+{'spring':'ST','night':'DT','multiple':'MT'}[target_environment(protocol,ds)]


def unpack_prompt(prompt):
    prefix, rest = prompt.split(MARKER, 1)
    examples, suffix = rest.split(END, 1)
    return prefix + MARKER, ast.literal_eval(examples), END + suffix


def archived_tof(archive,run):
    if 'tof' in run:return run['tof']
    # Older FR archive predates the run-record TOF field. Recover only counts
    # actually present in saved files, never invented intermediate diagnostics.
    matches=[sorted((archive/'artifacts/tof').glob(pattern)) for pattern in
             ('*_seq.pkl','*_seq_filter.pkl','*_seq_filter_true.pkl')]
    if any(len(paths)!=1 for paths in matches):raise ValueError('Ambiguous archived TOF files')
    return dict(zip(('input_count','first_pass_count','final_count'),
                    [len(load(paths[0])) for paths in matches]),
                provenance='counts read from existing archived TOF pickle files')


def preserved_gcad_state(config,gcad,artifact=None):
    """Preserve the archived auto decision; never silently enable guidance."""
    enabled=bool(gcad.get('guidance_enabled',False))
    legacy=False
    if not enabled:
        disabled=gcad.get('status')=='disabled' and gcad.get('relationship_count')==0
        legacy=(disabled and config.get('gcad_mode') is None and artifact is not None
            and bool(gcad.get('artifact_fingerprint'))
            and artifact.get('artifact_fingerprint')==gcad['artifact_fingerprint']
            and artifact.get('status')=='disabled' and artifact.get('lagged_behavior_relations')==[]
            and bool(artifact.get('disabled_reason')))
        if not ((config.get('gcad_mode')=='auto' and disabled) or legacy):
            raise ValueError('Unverified disabled GCAD state; do not silently treat as active guidance')
    state={'guidance_enabled':enabled,'archived_mode':config.get('gcad_mode'),
            'interpretation':('Original enabled GCAD guidance preserved' if enabled else
                'Original GCAD-auto pipeline disabled guidance; zero prompt edges. This is NOT an active-GCAD three-module comparison.')}
    if legacy:
        state.update(mode_field_missing_in_legacy_config=True,
            archived_disabled_reason=artifact['disabled_reason'],
            interpretation='Original legacy archive confirms disabled GCAD and zero prompt edges; no mode flag was recorded or inferred. NOT an active-GCAD three-module comparison.')
    return state


def baseline_detector_dir(spec):
    replay=spec.get('baseline_reevaluation')
    return Path(replay['detector_dir']) if replay else Path(spec['archive'])/'artifacts/anomaly_detection'


def verified_baseline_reevaluation(path,archive,original):
    """Explicit legacy protocol repair, never claim recovered historical splits."""
    saved=read(path)
    assert saved['new_codex_calls']==0 and saved['TOF_rerun'] is False
    assert saved['historical_split_recovered'] is False
    assert saved['original_record']==original
    final,=(archive/'artifacts/tof').glob('*_seq_filter_true.pkl')
    assert saved['input']==str(final) and saved['input_sha256']==sha(final)
    for file,digest in saved['files_sha256'].items():assert sha(file)==digest
    folder=Path(saved['detector_dir']);detection=saved['detection']
    training,validation=load(folder/'train.pkl'),load(folder/'validation.pkl')
    assert len(training)==detection['training_sequence_count']
    assert len(validation)==detection['validation_sequence_count']
    assert len(training)+len(validation)==original['generated_sequence_count']
    assert Counter(map(tuple,training+validation))==Counter(map(tuple,load(final)))
    return saved


def audit(args):
    destination = args.output / "protocol.json"
    if destination.exists():
        saved = read(destination)
        if saved.get('source_environment','winter')!=getattr(args,'source_env','winter'):
            raise ValueError('Output belongs to a different source environment; use an isolated directory')
        if saved["frozen_rasr_inputs"] != frozen_files(tuple(saved['datasets']),saved.get('source_environment','winter')):
            raise ValueError("RaSR algorithm or frozen data changed; do not resume")
        for spec in saved['datasets'].values():
            if 'baseline_reevaluation' in spec:
                assert verified_baseline_reevaluation(Path(spec['baseline_reevaluation_receipt']),
                    Path(spec['archive']),spec['reference_metrics'])==spec['baseline_reevaluation']
        return saved
    source_env=getattr(args,'source_env','winter')
    target_env={'winter':'spring','daytime':'night','single':'multiple'}[source_env]
    datasets=(args.dataset,) if source_env!='winter' or getattr(args,'dataset',None)=='fr' else ('sp','us')
    result = {"version": "formal-joint-ssc-v1", "seeds": list(SEEDS), "arms": list(ARMS),
        "frozen_rasr_inputs": frozen_files(datasets,source_env), "independent_regenerator": False,
        "selection": "joint I+J; original train-mode SSC; no origin quota",
        "ssc_fit": "same archived real-source-trained backbone shared by all arms and repetitions",
        "ssc_pool_order": "original parent order, each parent followed by its frozen variants; same layout for B1/B3",
        "budget": "historical actual Prompt category counts and per-category example counts, equal for all arms",
        "repetition": "independent fresh Codex requests plus downstream seeds 2024/2025/2026; frozen source/SSC/relations",
        "codex": {"model": "gpt-5.6-sol", "reasoning_effort": "none", "sampling_seed_available": False},
        "tof": {"epochs": 10, "implementation_unchanged": True},
        "detector": {"epochs": 15, "batch_size": 32, "learning_rate": .001,
            "train_ratio": .8, "validation_percentile": 95, "implementation_unchanged": True},
        "old_50_percent_control": str(ROOT / "rasr_research/run_experiment.py"),
        "no_test_based_tuning": True, "datasets": {}}
    if source_env!='winter':result['source_environment']=source_env
    if datasets==('fr',) or source_env!='winter':
        result.update(seeds=[2024],arms=['B3'],
            repetition='One explicitly approved B3 generation; historical B0 reused, no ablation or repeated generation')
    for ds in datasets:
        archive = list((args.source / "experiment_archive/formal").glob(f"{ds}_{source_env}_to_{target_env}_*"))
        if len(archive) != 1:
            raise ValueError(f"ambiguous archive {archive}")
        archive = archive[0]
        manifest = read(archive / "manifest.json")
        run = read(archive / "artifacts/generation/run_record/manifest.json")
        templates, day_budget = [], Counter()
        for cat in run["categories"]:
            file = archive / f"artifacts/generation/run_record/prompts/category_{cat}.txt"
            prefix, examples, suffix = unpack_prompt(file.read_text())
            day = int(str(cat).split("_")[0])
            day_budget[day] += len(examples)
            templates.append({"category": str(cat), "day": day, "examples": len(examples),
                              "path": str(file), "sha256": sha(file)})
        prepared = prepared_path(ds,source_env)
        original = args.source / f"SmartGen/IoT_data/{ds}/{source_env}/trn.pkl"
        pilot = read(prepared / "manifest.json")
        assert sha(original) == (pilot['input_sha256'][str(original)] if 'input_sha256' in pilot else pilot['source_sha256'])
        source_tss = args.source / f"SmartGen/IoT_data/{ds}/{source_env}/split_trn.pkl"
        assert sha(source_tss) == (pilot['input_sha256'][str(source_tss)] if 'input_sha256' in pilot else pilot['true_tss_sha256'])
        assert [r["row"] for r in load(prepared/"I.pkl")] == load(source_tss)
        equal_code = {}
        for relative in ("models1.py", "sppc.py", "find_categories.py", "security_check.py", "baseline1.py"):
            equal_code[relative] = sha(ROOT/"SmartGen"/relative) == sha(args.source/"SmartGen"/relative)
        assert all(equal_code.values())
        result["datasets"][ds] = {"archive": str(archive), "reference_config": manifest["config"],
            "reference_metrics": read(archive / "artifacts/anomaly_detection/metrics.json"),
            "categories": templates, "day_example_budget": {str(k):v for k,v in day_budget.items()},
            "calls_per_run": len(templates), "examples_per_run": sum(day_budget.values()),
            "historical_generation": {"tof": archived_tof(archive,run)}, "source": str(original),
            "source_sha256": sha(original), "real_tss_sha256": sha(source_tss),
            "ssc_checkpoint": str(archive / "artifacts/ssc/model.pth"),
            "ssc_checkpoint_sha256": sha(archive / "artifacts/ssc/model.pth"),
            "unchanged_core_code": equal_code,
            "detector_validation_percentile":manifest['config']['validation_percentile'],
            "pilot_differences": {"data": "same true source/target identity; isolated output paths",
                "ssc": "pilot eval mode + prefix truncation + forced 50% J; formal original train-mode joint greedy SSC",
                "gss": "same true-source GSS, never re-count J",
                "gcad": "same archived true-source GCAD; SP 2 / US balanced 12 prompt edges",
                "prompt": "SP pilot incorrectly inherited a later 4-6-event length constraint absent in historical SP; exact historical templates restored",
                "generation": f"pilot 2 calls / 8 examples -> {len(templates)} calls / {sum(day_budget.values())} examples, all seven days",
                "tof": "same 10-epoch algorithm; pilot only 8-9 inputs vs historical full-size inputs",
                "detector": "same 15-epoch architecture, Adam, 80/20 split, seed protocol; no tiny pilot dataset cap",
                "threshold": "same p95 of generated validation losses, not calibrated on target test labels"}}
        if ds=='fr':
            result['datasets'][ds]['pilot_differences']['gcad']='Same original archived true-source FR GCAD and prompt wording'
            result['datasets'][ds]['pilot_differences']['threshold']='Original FR p95.5 generated-validation protocol, not US/SP p95'
        if source_env!='winter':
            spec=result['datasets'][ds]
            assert manifest['config']['original_environment']==source_env
            assert manifest['config']['target_environment']==target_env
            gcad_artifact=None
            if manifest['config'].get('gcad_mode') is None and run['gcad'].get('status')=='disabled':
                hints=archive/'artifacts/gcad/gcad_hints.json'
                gcad_artifact=read(hints)
                assert gcad_artifact['source']['sha256']==sha(source_tss)
                assert gcad_artifact['source']['dataset']==ds
                spec['legacy_disabled_gcad_artifact_sha256']=sha(hints)
            spec['gcad_comparison_state']=preserved_gcad_state(manifest['config'],run['gcad'],gcad_artifact)
            historical=spec['reference_metrics']
            if historical['training_sequence_count']+historical['validation_sequence_count']!=historical['generated_sequence_count']:
                if source_env!='single':raise ValueError('Unverified historical split')
                receipt=args.output/f'baseline-reevaluation/{ds}/result.json'
                spec['baseline_reevaluation']=verified_baseline_reevaluation(receipt,archive,historical)
                spec['baseline_reevaluation_receipt']=str(receipt)
                spec['historical_protocol_caveat']='Historical detector split unavailable and counts inconsistent; compare B3 to explicit existing-data baseline reevaluation for matched protocol. Original historical score is retained but is NOT an apples-to-apples control.'
            spec['historical_gcad']=run['gcad']
            spec['prepared_source']=str(prepared)
            spec['pilot_differences']={'scope':'New source task; no pilot results reused',
                'data':'Same true source TSS and target evaluation identity; isolated outputs',
                'ssc':'Original joint I+J greedy SSC, no origin quota; identical archived source checkpoint and threshold',
                'gss_gcad_prompt':'Exact archived prompt outside examples; no recounting on J',
                'budget':'Identical historical calls and per-category example upper bounds; actual SSC output, no refill',
                'tof_detector':'Unchanged implementation, historical percentile and original environment-specific token field'}
    atomic_write_json(destination, result)
    return result


def text_records(records, ds):
    maps = [dictionary.dayofweek_dict, dictionary.hour_dict, getattr(dictionary, f"{ds}_devices_dict"),
            getattr(dictionary, f"{ds}_actions")]
    inverse = [{v:k for k,v in m.items()} for m in maps]
    return [[inverse[i%4].get(v,str(v)) for i,v in enumerate(r["row"])] for r in records]


def baseline_selection(protocol, ds):
    original = load(ROOT / f"rasr_research/runs/pilot-20261124/{ds}/prepared/I.pkl")
    by_text = defaultdict(list)
    for r, t in zip(original, text_records(original, ds)):
        by_text[tuple(t)].append(r)
    selected = []
    for template in protocol["datasets"][ds]["categories"]:
        _, text, _ = unpack_prompt(Path(template["path"]).read_text())
        chosen = []
        for row in text:
            if not by_text[tuple(row)]:
                raise ValueError("archived SSC example not found in the unchanged real source")
            chosen.append(by_text[tuple(row)][0])
        selected.append(chosen)
    return selected


def original_greedy(vectors, threshold):
    """Same original greedy cosine criterion, without allocating N x N matrix."""
    vectors = normalize(vectors, copy=False)
    removed = np.zeros(len(vectors), dtype=bool)
    kept = []
    for i in range(len(vectors)):
        if removed[i]:
            continue
        kept.append(i)
        if i+1 < len(vectors):
            removed[i+1:] |= (vectors[i+1:] @ vectors[i]) > threshold
    return kept


def category_partition(by_day, templates):
    """Partition the actual SSC prefix; caps are not sample-count targets.

    Keep all historical calls, reserving one retained example per remaining
    call when possible. Never refill from rejected rows or duplicate a row.
    A genuinely empty category is recorded as empty, not silently padded.
    """
    remaining_calls = Counter(t["day"] for t in templates)
    offsets = Counter()
    selected = []
    for template in templates:
        day = template["day"]
        available = len(by_day[day]) - offsets[day]
        remaining_calls[day] -= 1
        reserve = min(remaining_calls[day], max(0, available - 1))
        count = min(template["examples"], max(0, available - reserve))
        selected.append(by_day[day][offsets[day]:offsets[day] + count])
        offsets[day] += count
    return selected


def budget_amendment(args):
    """Keep the initial protocol intact and record the explicit user decision."""
    atomic_write_json(args.output / "budget-amendment.json", {
        "decision_date": "2026-11-24", "authority": "explicit human reply in this conversation",
        "user_reply": "调用预算一致，记录实际示例数（推荐）",
        "overrides": "protocol.json budget: per-category counts are upper bounds, not required counts",
        "policy": "identical historical call counts and example upper budgets; actual SSC output without refill",
        "empty_category": "retain call with empty examples; explicitly record actual count zero",
        "unchanged": ["RaSR C/T/S", "insert/replace algorithms", "SSC model and threshold",
                      "GSS", "GCAD", "Codex model", "TOF", "detector", "threshold calibration"]})


def compress(args, protocol, ds, arm):
    output = args.output / ds / "selection" / arm
    receipt = output / "manifest.json"
    if receipt.exists():
        return load(output / "categories.pkl")
    output.mkdir(parents=True, exist_ok=True)
    if arm == "B0":
        selected = baseline_selection(protocol, ds)
        atomic_pickle_dump(output/"categories.pkl", selected)
        atomic_write_json(receipt, {"mode": "verified actual archived SSC examples", "J_input": 0,
            "J_selected": 0, "actual_examples": sum(map(len,selected))})
        return selected
    prepared = Path(protocol['datasets'][ds].get('prepared_source',prepared_path(ds)))
    originals, variants = load(prepared/"I.pkl"), load(prepared/f"{arm}_J.pkl")
    linked = defaultdict(list)
    for r in variants:
        linked[r["parent_id"]].append(r)
    pool = [r for parent in originals for r in [parent, *linked[parent["id"]]]]
    atomic_write_json(output/"input.json", {"mode":"I+J joint", "counts":dict(Counter(r["origin"] for r in pool)),
        "input_ids_sha256":hashlib.sha256(json.dumps([r["id"] for r in pool]).encode()).hexdigest(),
        "frozen_variants_sha256": sha(prepared/f"{arm}_J.pkl"), "origin_quota": None})
    threshold = protocol["datasets"][ds]["reference_config"]["threshold"]
    vocab = vocabulary_size(ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(2024)  # identical fixed source-compression protocol for paired arms
    by_day, audits = {}, []
    for day in range(7):
        candidates = [r for r in pool if r["row"][0]==day]
        model = TransformerAutoencoder(vocab,d_model=256,nhead=4,num_encoder_layers=2,num_decoder_layers=2).to(device)
        model.load_state_dict(torch.load(protocol["datasets"][ds]["ssc_checkpoint"],map_location=device))
        model.train()  # restore original SSC dropout semantics, NOT pilot eval()
        source=torch.full((len(candidates),40),vocab-1,dtype=torch.long,device=device)
        for i,r in enumerate(candidates):
            if len(r["row"])>40: raise ValueError("source exceeds original SSC length")
            source[i,:len(r["row"])]=torch.tensor(r["row"],device=device)
        # No gradient is needed to select examples. Dropout stays ON as upstream.
        # Full day forward, not microbatching, preserves original RNG draw shape.
        with torch.no_grad():
            memory=model(source,src_key_padding_mask=source.eq(vocab-1))
            vectors=memory.cpu().numpy().reshape(len(candidates),-1)
        del memory,source,model
        torch.cuda.empty_cache()
        kept=original_greedy(vectors,threshold)
        all_kept=[candidates[i] for i in kept]
        atomic_pickle_dump(output/f"day-{day}-all-retained.pkl",all_kept)
        quota=int(protocol["datasets"][ds]["day_example_budget"][str(day)])
        by_day[day]=all_kept[:quota]
        audits.append({"day":day,"input":len(candidates),"all_retained":len(all_kept),"prompt_budget":quota,
            "actual_prompt_examples":len(by_day[day]),"unfilled_example_budget":max(0,quota-len(all_kept)),
            "all_retained_J":sum(r["origin"]=="J" for r in all_kept),
            "prompt_J":sum(r["origin"]=="J" for r in by_day[day])})
        print(f"SSC {ds} {arm} day={day} input={len(candidates)} retained={len(all_kept)} budget={quota} J={audits[-1]['prompt_J']}",flush=True)
        del vectors,candidates,all_kept
        gc.collect()
    selected=category_partition(by_day,protocol["datasets"][ds]["categories"])
    atomic_pickle_dump(output/"categories.pkl",selected)
    atomic_write_json(receipt,{"mode":"original SSC on full joint I+J; fixed calls and example upper budget",
        "origin_quota":None,"source_checkpoint_sha256":protocol["datasets"][ds]["ssc_checkpoint_sha256"],
        "J_input":len(variants),"J_selected_for_prompts":sum(r["origin"]=="J" for c in selected for r in c),
        "days":audits,"actual_examples":sum(map(len,selected)),"semantics":"memory-only optimization: no_grad and row-wise greedy cosine; no algorithm/threshold change"})
    return selected


def queue(args, protocol, ds, arm, seeds=SEEDS):
    selected=compress(args,protocol,ds,arm)
    spec=protocol["datasets"][ds]
    for seed in seeds:
        run=args.output/ds/f"seed-{seed}"/arm
        for index,(template,records) in enumerate(zip(spec["categories"],selected)):
            directory=run/f"codex/call-{index:02d}"
            prefix,old,suffix=unpack_prompt(Path(template["path"]).read_text())
            examples=text_records(records,ds)
            assert len(examples)<=template["examples"]
            prompt=prefix+str(examples)+suffix
            if arm=="B0":
                assert examples==old
                assert prompt==Path(template["path"]).read_text(),"B0 prompt must reproduce historical template and examples exactly"
            directory.mkdir(parents=True,exist_ok=True)
            file=directory/"prompt.txt"
            if file.exists() and file.read_text()!=prompt: raise ValueError("queued prompt changed")
            atomic_write_text(file,prompt)
            atomic_write_json(directory/"input_audit.json",{"category":template["category"],"seed":seed,"arm":arm,
                "source_template_sha256":template["sha256"],"actual_examples":len(records),
                "example_upper_budget":template["examples"],
                "origins":dict(Counter(r["origin"] for r in records)),"records":records,
                "text_examples":examples,"prompt_sha256":sha(file)})
            atomic_write_json(directory/"request.json",{"model":"gpt-5.6-sol","reasoning_effort":"none",
                "prompt_sha256":sha(file),"priority":[ARMS.index(arm),seed, index,ds]})
        atomic_write_json(run/"status.json",{"status":"queued","categories":len(selected),"seed":seed,
            "actual_prompt_examples":sum(map(len,selected)),"example_upper_budget":spec["examples_per_run"],
            "J_prompt_examples":sum(r["origin"]=="J" for c in selected for r in c),"selection_shared_frozen":True})
    print(f"QUEUED {ds} {arm}: {len(selected)} calls x {len(seeds)} repetition(s); no remote calls made",flush=True)


def target_copies(args, ds, runtime,environment='spring'):
    normal_rel,attack_rel=test_relatives(ds,environment)
    verified=args.output/f'historical-checkpoint-score/{ds}/result.json'
    if verified.exists():
        previous=read(verified)
        assert sha(args.source/'SmartGen'/normal_rel)==previous['normal_test_sha256']
        assert sha(args.source/'SmartGen'/attack_rel)==previous['attack_test_sha256']
    for rel in (normal_rel,attack_rel):
        dest=runtime/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.source/"SmartGen"/rel,dest)


def numeric_sequence(sequence, maps):
    """Trace existing Transnum conversion, without changing its output/filter."""
    converted=[]
    for index,text in enumerate(sequence):
        mapping=maps[index%4]
        key=text
        if key not in mapping and index%4==3:
            key=sequence[index-1]+':'+text
        converted.append(mapping.get(key,99999))
    return [value for offset in range(0,len(converted),4)
            if 99999 not in converted[offset:offset+4] for value in converted[offset:offset+4]]


def recover_saved_response(directory):
    """Recover real, complete stdout without another model request.

    The previous Windows GBK stderr reader crashed after stdout was saved but
    before the exit receipt was written. Never invent an exit code or stderr.
    """
    directory=Path(directory)
    if (directory/'response.txt').exists():return False
    error=read(directory/'worker-error.json')
    if error['type']!='TypeError' or error['error']!='data must be str, not NoneType':
        raise ValueError('Only the audited Windows pipe reader failure may be recovered here')
    request=read(directory/'request.json')
    assert sha(directory/'prompt.txt')==request['prompt_sha256']
    raw=(directory/'stdout.txt').read_text(encoding='utf-8')
    block=CodexClient.extract_sequence_block(raw)
    parsed=CodexClient.parse_sequence_block(block)
    if not parsed:raise ValueError('Saved stdout has no generated sequences')
    assert not (directory/'exit.json').exists(),'Do not overwrite an original exit receipt'
    atomic_write_json(directory/'exit.json',{'returncode':None,
        'recovered_from_saved_stdout':True,'original_cli_returncode_unavailable':True,
        'stdout_sha256':sha(directory/'stdout.txt'),'prompt_sha256':request['prompt_sha256'],
        'model':request['model'],'reasoning_effort':request['reasoning_effort'],
        'parsed_sequences':len(parsed),'new_model_calls':0,
        'cause':'Windows GBK stderr decoding error followed by audit NoneType write error',
        'missing_stderr_not_reconstructed':True})
    atomic_write_text(directory/'response.txt',block)
    return True


def verify_response_receipt(directory):
    receipt=read(directory/'exit.json')
    if receipt['returncode']==0:return receipt
    if receipt['returncode'] is None and receipt.get('recovered_from_saved_stdout'):
        assert receipt['original_cli_returncode_unavailable'] is True
        assert receipt['stdout_sha256']==sha(directory/'stdout.txt')
        assert receipt['prompt_sha256']==sha(directory/'prompt.txt')
        raw=(directory/'stdout.txt').read_text(encoding='utf-8')
        assert CodexClient.extract_sequence_block(raw)==(directory/'response.txt').read_text()
        assert CodexClient.parse_sequence_block((directory/'response.txt').read_text())
        return receipt
    raise ValueError('Nonzero or unverified CLI outcome: do not evaluate this response')


def evaluate_run(args, protocol, ds, seed, arm):
    environment=target_environment(protocol,ds)
    run=args.output/ds/f"seed-{seed}"/arm
    if (run/"result.json").exists(): return read(run/"result.json")
    templates=protocol["datasets"][ds]["categories"]
    calls=[run/f"codex/call-{i:02d}" for i in range(len(templates))]
    if not all((p/"response.txt").exists() for p in calls): return None
    runtime=run/"runtime"; runtime.mkdir(exist_ok=True)
    target_copies(args,ds,runtime,environment)
    threshold=protocol["datasets"][ds]["reference_config"]["threshold"]
    label=f"formal-{arm}-s{seed}"
    categories=[t["category"] for t in templates]
    target=runtime/f"IoT_data/{ds}/{environment}"; target.mkdir(parents=True,exist_ok=True)
    (runtime/f"filter_data/{ds}/{environment}").mkdir(parents=True,exist_ok=True)
    evidence=[]
    for t,p in zip(templates,calls):
        receipt=verify_response_receipt(p)
        response=(p/"response.txt").read_text(); parsed=CodexClient.parse_sequence_block(response)
        atomic_pickle_dump(target/f"{ds}_{environment}_generation_day_{t['category']}_SPPC_th={threshold}_{label}.pkl",response)
        audit=read(p/"input_audit.json"); assert sha(p/"prompt.txt")==audit["prompt_sha256"]
        evidence.append({"category":t["category"],"prompt_sha256":audit["prompt_sha256"],
            "cli_returncode":receipt['returncode'],
            "recovered_from_saved_stdout":receipt.get('recovered_from_saved_stdout',False),
            "actual_examples":audit["actual_examples"],"example_upper_budget":t["examples"],
            "J_ids":[r["id"] for r in audit["records"] if r["origin"]=="J"],
            "response_sha256":sha(p/"response.txt"),"parsed_sequences":len(parsed),
            "provenance_not_causal_attribution":True})
    atomic_write_json(run/"status.json",{"status":"TOF-and-detection","seed":seed})
    maps=[dictionary.dayofweek_dict,dictionary.hour_dict,getattr(dictionary,f"{ds}_devices_dict"),getattr(dictionary,f"{ds}_actions")]
    with (run/"evaluation.log").open("w",encoding="utf-8",buffering=1) as log,redirect_stdout(log),redirect_stderr(log),cwd(runtime):
        Extract(ds,environment,threshold,"SPPC",label,categories)
        Transnum(ds,environment,threshold,"SPPC",label,categories,maps)
        tof=security_check(ds,environment,threshold,"SPPC",label,seed=seed,keep_intermediates=True)
        percentile=protocol['datasets'][ds].get('detector_validation_percentile',95)
        detection=Anomaly_detection(ds,environment,threshold,"SPPC",label,percentile,seed=seed)
    raw=load(runtime/f"filter_data/{ds}/{environment}/{ds}_{environment}_generation_SPPC_th={threshold}_{label}_seq.pkl")
    final=load(runtime/tof["output_path"])
    final_counter=Counter(tuple(r) for r in final)
    detector_dir=runtime/Path(detection["metrics_path"]).parent
    train_counter=Counter(tuple(r) for r in load(detector_dir/'train.pkl'))
    validation_counter=Counter(tuple(r) for r in load(detector_dir/'validation.pkl'))
    converted_all=[]
    for entry,t in zip(evidence,templates):
        seqs=load(target/f"{ds}_{environment}_generation_day_{t['category']}_SPPC_th={threshold}_{label}_seq.pkl")
        entry["generated_sequences"]=[]
        for i,text in enumerate(seqs):
            numeric=numeric_sequence(text,maps)
            if numeric:converted_all.append(numeric)
            entry["generated_sequences"].append({"generated_id":f"{ds}:{seed}:{arm}:{t['category']}:{i}",
                "raw_text":text,"converted_numeric":numeric,
                "numeric_equivalence_class_sha256":hashlib.sha256(json.dumps(numeric).encode()).hexdigest(),
                "matching_TOF_final_copies":final_counter[tuple(numeric)],
                "matching_detector_train_copies":train_counter[tuple(numeric)],
                "matching_detector_validation_copies":validation_counter[tuple(numeric)],
                "duplicate_attribution":"matching counts refer to identical numeric rows, not unique causality"})
        entry["conversion_output"]=str(runtime/f"filter_data/{ds}/{environment}/{ds}_{environment}_generation_SPPC_th={threshold}_{label}_seq.pkl")
    assert converted_all==raw,"evidence trace must agree exactly with existing Transnum output"
    result={"status":"completed","dataset":ds,"arm":arm,"seed":seed,"tof":tof,"detection":detection,
        "actual_codex_calls":len(calls),"prompt_examples":sum(e["actual_examples"] for e in evidence),
        "prompt_example_upper_budget":sum(t["examples"] for t in templates),
        "actual_J_prompt_examples":sum(len(e["J_ids"]) for e in evidence),
        "evidence":evidence,"note":"J -> SSC -> prompt -> response lineage is observed; individual generated actions cannot be causally attributed to J from logs alone."}
    atomic_write_json(run/"result.json",result)
    atomic_write_json(run/"status.json",{"status":"completed","seed":seed,"metrics":detection["metrics"]})
    print(f"COMPLETED {ds} {arm} seed={seed} F1={detection['metrics']['f1_score']:.6f} n={tof['final_count']}",flush=True)
    gc.collect(); torch.cuda.empty_cache()
    return result


def replay(args, protocol, ds):
    """Historical-input replay, explicitly NOT an independent fresh generation."""
    run=args.output/"historical-replay"/ds
    if (run/"result.json").exists(): return
    run.mkdir(parents=True,exist_ok=True); runtime=run/"runtime";runtime.mkdir(exist_ok=True)
    spec=protocol["datasets"][ds]; archive=Path(spec["archive"])
    threshold=spec["reference_config"]["threshold"];label="historical-replay"
    directory=runtime/f"filter_data/{ds}/spring";directory.mkdir(parents=True,exist_ok=True)
    candidates=[p for p in (archive/"artifacts/tof").glob("*_seq.pkl")]
    assert len(candidates)==1
    shutil.copy2(candidates[0],directory/f"{ds}_spring_generation_SPPC_th={threshold}_{label}_seq.pkl")
    target_copies(args,ds,runtime)
    with (run/"evaluation.log").open("w",encoding="utf-8",buffering=1) as log,redirect_stdout(log),redirect_stderr(log),cwd(runtime):
        tof=security_check(ds,"spring",threshold,"SPPC",label,seed=2024,keep_intermediates=True)
        detection=Anomaly_detection(ds,"spring",threshold,"SPPC",label,95,seed=2024)
    atomic_write_json(run/"result.json",{"mode":"historical generated input replay; zero new Codex calls; not counted among 3 repeats",
        "source_sha256":sha(candidates[0]),"reference":spec["reference_metrics"],"tof":tof,"detection":detection})
    print(f"HISTORICAL REPLAY {ds}: F1={detection['metrics']['f1_score']}",flush=True)


def summary(args, protocol):
    records=[];aggregate={}
    for ds in ("sp","us"):
        aggregate[ds]={}
        for arm in ARMS:
            values=[]
            for seed in SEEDS:
                path=args.output/ds/f"seed-{seed}"/arm/"result.json"
                if path.exists(): values.append(read(path));records.append(read(path))
            aggregate[ds][arm]={"completed_repetitions":len(values)}
            if values:
                aggregate[ds][arm]["metrics"]={key:{"mean":float(np.mean([r["detection"]["metrics"][key] for r in values])),
                    "std":float(np.std([r["detection"]["metrics"][key] for r in values],ddof=1)) if len(values)>1 else None}
                    for key in ("f1_score","precision","recall","FPR","FNR","auroc","auprc")}
    atomic_write_json(args.output/"summary.json",{"completed":len(records),"required":18,"results":records,
        "aggregate":aggregate,"all_results_retained":True,"test_based_tuning":False})


def single_dataset(args):
    return getattr(args,'dataset',None) or 'sp'


def single_file(args, ds, stem):
    # Preserve completed SP filenames byte-for-byte; US has its own artifacts.
    return args.output/(f'{stem}.json' if ds=='sp' else f'{stem}-{ds}.json')


def prepare_single(args, protocol):
    """Latest human scope supersedes the retired 18-run schedule."""
    ds=single_dataset(args)
    policy_path = ROOT/'rasr_research/generation-policy.json'
    if not policy_path.exists():
        raise ValueError("Explicit single-run generation policy is required before queue preparation")
    policy = read(policy_path)
    assert args.output.resolve()==(policy_path.parent/policy['queue_root']).resolve(), 'Queue output must match the single authorized scenario root'
    assert policy['allowed_runs'] == [f'{ds}/seed-2024/B3']
    spec = protocol['datasets'][ds]
    assert len(spec['categories']) == policy['max_additional_calls']
    archive = Path(spec['archive'])
    detector=baseline_detector_dir(spec)
    files = [archive/'artifacts/anomaly_detection/metrics.json',
             detector/'train.pkl',
             detector/'validation.pkl',
             archive/'artifacts/anomaly_detection/detector.pth',
             archive/'artifacts/gss/action_transitions.json',
             archive/'artifacts/gcad/gcad_hints.json',
             archive/'artifacts/ssc/model.pth',
             *sorted((archive/'artifacts/tof').glob('*_seq.pkl')),
             *sorted((archive/'artifacts/tof').glob('*_seq_filter_true.pkl'))]
    baseline = read(files[0])
    assert baseline == spec['reference_metrics']
    actual=spec.get('baseline_reevaluation',{}).get('detection',baseline)
    assert len(load(detector/'train.pkl')) == actual['training_sequence_count']
    assert len(load(detector/'validation.pkl')) == actual['validation_sequence_count']
    if 'baseline_reevaluation' in spec:
        files.extend([Path(spec['baseline_reevaluation_receipt']),detector/'metrics.json',detector/'detector.pth'])
    queue(args, protocol, ds, 'B3', seeds=(2024,))
    prompt_checks = []
    run = args.output/f'{ds}/seed-2024/B3'
    for index, template in enumerate(spec['categories']):
        path = run/f'codex/call-{index:02d}/prompt.txt'
        old_prefix, _, old_suffix = unpack_prompt(Path(template['path']).read_text())
        prefix, examples, suffix = unpack_prompt(path.read_text())
        assert (prefix, suffix) == (old_prefix, old_suffix)
        assert len(examples) <= template['examples']
        prompt_checks.append({'category':template['category'],'actual_examples':len(examples),
                              'example_upper_budget':template['examples'],'non_example_content_identical':True})
    atomic_write_json(single_file(args,ds,'active-scope'), {
        'version':f'single-{scenario(protocol,ds)}-B3-historical-baseline-v1','dataset':ds,'arm':'B3','seed':2024,
        'runs_required':1,'historical_baseline':baseline,
        'baseline_reevaluation':spec.get('baseline_reevaluation'),
        'historical_protocol_caveat':spec.get('historical_protocol_caveat'),
        'baseline_files_sha256':{str(p):sha(p) for p in files},
        'baseline_new_codex_calls':0,'planned_B3_calls':len(spec['categories']),'repetitions':1,
        'baseline_selection_reason':'complete original formal archive; scenario fixed before seeing B3 results',
        'supersedes':'protocol.json 18-run schedule; the old protocol is retained as historical evidence only',
        'suspended':['all fresh B0','B1','B2','B4','B5','other scenarios','seeds 2025/2026','independent regenerator'],
        'generation_policy':str(policy_path),'same_formal_tof':protocol['tof'],
        'same_formal_detector':dict(protocol['detector'],validation_percentile=spec.get('detector_validation_percentile',95)),'prompt_checks':prompt_checks,
        'ssc':f"original joint I+J selection, no forced J ratio; original source checkpoint and threshold {spec['reference_config']['threshold']}",
        'limits':'same protocol is not bitwise historical reproduction; Codex sampling seed unavailable; only one B3 run, no significance claim'})
    single_report(args, protocol)


def explicit_resume_accounting(run,current_attempts):
    receipt=run/'explicit-resume-20261125.json'
    if not receipt.exists():return {}
    resumed=read(receipt)
    for call,digest in resumed['completed_slots_reused'].items():
        assert sha(run/'codex'/call/'response.txt')==digest,'Never regenerate or replace completed responses on resume'
    saved_failures=list((run/'codex').glob('call-*/prior-failed-attempt-*/worker-error.json'))
    assert len(saved_failures)==resumed['prior_failed_attempts']
    return {'saved_prior_failed_CLI_attempts':len(saved_failures),
        'total_CLI_attempts_including_saved_failures':current_attempts+len(saved_failures),
        'explicit_resume_receipt':str(receipt),'explicit_resume_receipt_sha256':sha(receipt),
        'CLI_internal_reconnects_not_counted_as_new_worker_attempts':True}


def single_report(args, protocol):
    ds=single_dataset(args)
    spec=protocol['datasets'][ds]
    report_path=single_file(args,ds,'single-b3-comparison')
    run = args.output/f'{ds}/seed-2024/B3'
    calls = sorted((run/'codex').glob('call-*'))
    completed = sum((p/'response.txt').exists() for p in calls)
    errors = [str(p/'worker-error.json') for p in calls if (p/'worker-error.json').exists() and not (p/'response.txt').exists()]
    audits = [read(p/'input_audit.json') for p in calls]
    result_path = run/'result.json'
    result = read(result_path) if result_path.exists() else None
    baseline = spec['reference_metrics']
    policy = read(ROOT/'rasr_research/generation-policy.json')
    rescore_path = args.output/f'historical-checkpoint-score/{ds}/result.json'
    rescore = read(rescore_path) if rescore_path.exists() else None
    selection=read(args.output/f'{ds}/selection/B3/manifest.json')
    source_input=read(args.output/f'{ds}/selection/B3/input.json')
    comparable_baseline=dict(baseline['metrics'])
    if rescore and rescore['matches_original_recorded_metrics']:
        comparable_baseline.update({k:rescore['metrics'][k] for k in ('auroc','auprc')})
    metrics = result['detection']['metrics'] if result else None
    prior_baseline_path=args.output/f'{ds}/seed-2024/B0/result.json'
    prior_baseline=read(prior_baseline_path) if prior_baseline_path.exists() else None
    delta = {key:metrics[key]-value for key,value in comparable_baseline.items()
             if metrics and key in metrics and key in ('f1_score','precision','recall','FPR','FNR','auroc','auprc')}
    atomic_write_json(report_path, {
        'status':'completed' if result else ('blocked-generation' if errors or policy.get('hold') else 'awaiting-generation'),
        'scenario':scenario(protocol,ds),'baseline_type':('original project GCAD-disabled formal archive, not newly generated B0'
            if spec.get('gcad_comparison_state',{}).get('guidance_enabled') is False else
            'original project formal Gen+GCAD archive, not newly generated B0'),
        'gcad_comparison_state':spec.get('gcad_comparison_state'),
        'historical_baseline':baseline,'B3_metrics':metrics,'B3_minus_historical':delta,
        'comparison_baseline_metrics':comparable_baseline,
        'ranking_baseline_provenance':'original archived detector re-scored at original threshold; threshold metrics verified identical' if rescore and rescore['matches_original_recorded_metrics'] else None,
        'B3_result_path':str(result_path) if result else None,
        'generation':{'required_calls':len(spec['categories']),'valid_responses':completed,'failure_logs':errors,
                      'recovered_saved_stdout_calls':[p.name for p in calls if (p/'exit.json').exists() and read(p/'exit.json').get('recovered_from_saved_stdout')],
                      'started_current_attempts':sum((p/'worker-claim.json').exists() for p in calls),
                      'new_baseline_calls':0,
                      'new_calls_by_this_prepare_or_evaluate_command':0,'automatic_retries':False,
                      'retry_scope':'worker-level regeneration disabled; original CLI internal reconnect/fallback unchanged and audited in stderr.txt'},
        'prompt':{'actual_examples':sum(a['actual_examples'] for a in audits),
                  'upper_budget':spec['examples_per_run'],'J_examples':sum(a['origins'].get('J',0) for a in audits),'forced_J_quota':None},
        'data_volumes':{'real_source_to_SSC':source_input['counts'].get('I',0),
            'J_to_SSC':selection['J_input'],
            'J_retained_by_SSC_before_prompt_cap':sum(d['all_retained_J'] for d in selection['days']),
            'historical_generated_before_TOF':spec['historical_generation']['tof']['input_count'],
            'historical_TOF_final':baseline['generated_sequence_count'],
            'B3_Codex_parsed_sequences':sum(e['parsed_sequences'] for e in result['evidence']) if result else None,
            'B3_generated_before_TOF':result['tof']['input_count'] if result else None,
            'B3_TOF_final':result['tof']['final_count'] if result else None,
            'B3_detector_training':result['detection']['training_sequence_count'] if result else None,
            'B3_detector_validation':result['detection']['validation_sequence_count'] if result else None},
        'threshold_protocol':f"same generated-validation p{spec.get('detector_validation_percentile',95):g} protocol, not the same numeric detector threshold across separately trained models",
        'historical_auroc_auprc':'not recorded in original metrics; do not invent or substitute replay scores',
        'historical_checkpoint_rescore':rescore,
        'archived_prior_failure_logs':[str(p) for p in (run/'codex').glob('call-*/prior-failed-attempt-*/worker-error.json')],
        'hold_reason':policy.get('hold_reason') if not result else None,
        'conclusion':'B3 has no completed formal result; cannot judge improvement' if not result else
                     'single-run historical comparison, not evidence of stable causal improvement',
        'previous_results_retained_but_excluded_from_primary_comparison':True})
    report=read(report_path)
    report['generation'].update(explicit_resume_accounting(run,report['generation']['started_current_attempts']))
    if 'baseline_reevaluation' in spec:
        replay=spec['baseline_reevaluation']
        report['historical_protocol_caveat']=spec['historical_protocol_caveat']
        report['matched_protocol_baseline']=replay['detection']
        report['baseline_reevaluation_receipt']=spec['baseline_reevaluation_receipt']
        report['B3_minus_matched_protocol_baseline']={key:metrics[key]-value
            for key,value in replay['detection']['metrics'].items()
            if metrics and key in metrics and key in ('f1_score','precision','recall','FPR','FNR','auroc','auprc')}
    report['previous_scope_existing_B0_not_rerun']={
        'path':str(prior_baseline_path),'metrics':prior_baseline['detection']['metrics'],
        'TOF_final':prior_baseline['tof']['final_count'],
        'included_in_primary_comparison':False,'new_generation_or_training_in_this_resume':False,
        'interpretation':'Previously completed under superseded instructions; retained transparently. Its higher/lower score must not be hidden or treated as a new experiment in the current scope.'
    } if prior_baseline else None
    atomic_write_json(report_path,report)
    if not result:
        atomic_write_json(run/'status.json', {'status':'blocked-generation' if errors or policy.get('hold') else 'awaiting-generation',
            'dataset':ds,'arm':'B3','seed':2024,'valid_responses':completed,
            'required_responses':len(spec['categories']),'hold_reason':policy.get('hold_reason'),'automatic_retries':False})


def score_historical_checkpoint(args, protocol):
    """Score the existing original detector: zero generation or training calls."""
    ds=single_dataset(args)
    output=args.output/f'historical-checkpoint-score/{ds}'
    receipt=output/'result.json'
    if receipt.exists():return
    output.mkdir(parents=True,exist_ok=True)
    reference=protocol['datasets'][ds]['reference_metrics']
    checkpoint=Path(protocol['datasets'][ds]['archive'])/'artifacts/anomaly_detection/detector.pth'
    fingerprint=sha(checkpoint)
    environment=target_environment(protocol,ds)
    normal_rel,attack_rel=test_relatives(ds,environment)
    attack=args.source/'SmartGen'/attack_rel
    normal=args.source/'SmartGen'/normal_rel
    setup_seed(2024)
    with (output/'evaluation.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
        evaluation=evaluate_detector(environment,vocabulary_size(ds),str(attack),str(normal),str(checkpoint),10,
            threshold=reference['detection_threshold'],scores_path=output/'continuous_scores.npz',detailed=True)
    names=('TP','TN','FP','FN','FPR','FNR','recall','precision','accuracy','f1_score')
    metrics={name:float(value) for name,value in zip(names,evaluation['threshold_metrics'])}
    metrics.update({name:evaluation[name] for name in ('auroc','auprc')})
    matches=all(np.isclose(metrics[name],value,rtol=0,atol=1e-12) for name,value in reference['metrics'].items())
    assert sha(checkpoint)==fingerprint,'historical checkpoint must stay read-only'
    atomic_write_json(receipt,{'mode':'existing original formal detector checkpoint only; zero training or generation',
        'checkpoint':str(checkpoint),'checkpoint_sha256':fingerprint,
        'normal_test_sha256':sha(normal),'attack_test_sha256':sha(attack),
        'threshold':reference['detection_threshold'],'threshold_unchanged':True,
        'matches_original_recorded_metrics':bool(matches),'metrics':metrics,
        'historical_record_unchanged':True,'parameter_selection_or_tuning':False})
    print(f"HISTORICAL CHECKPOINT SCORE: matches={matches} F1={metrics['f1_score']:.6f}; zero Codex calls",flush=True)
    single_report(args,protocol)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path('/home/heyang/projects/SmartGen_GCAD'))
    parser.add_argument("--output",type=Path,default=ROOT/'rasr_research/runs/formal-20261124')
    parser.add_argument("--action",choices=['audit','prepare-single','evaluate-single','score-historical','wait-single','recover-stdout','status'],required=True,
                        help='Old multi-arm queue/watch actions are intentionally disabled by the latest user instruction')
    parser.add_argument("--dataset",choices=['sp','us','fr'])
    parser.add_argument('--source-env',choices=['winter','daytime','single'],default='winter')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); protocol=audit(args)
    if args.action!='audit': budget_amendment(args)
    if args.action=='audit': print(json.dumps(protocol,indent=2));return
    ds=single_dataset(args)
    active_policy=read(ROOT/'rasr_research/generation-policy.json')
    if active_policy['allowed_runs']!=[f'{ds}/seed-2024/B3'] and not (args.output/f'{ds}/seed-2024/B3/result.json').exists():
        parser.error('Only the explicitly authorized single B3 scenario may be run')
    if args.action=='prepare-single':prepare_single(args,protocol)
    if args.action=='status':single_report(args,protocol)
    if args.action=='recover-stdout':
        run=args.output/f'{ds}/seed-2024/B3'
        count=0
        for directory in sorted((run/'codex').glob('call-*')):
            if (directory/'worker-error.json').exists() and not (directory/'response.txt').exists():
                count+=recover_saved_response(directory)
        single_report(args,protocol)
        print(f'RECOVERED {count} saved stdout responses; zero model calls; unknown original exit status explicitly recorded',flush=True)
    if args.action=='score-historical':score_historical_checkpoint(args,protocol)
    if args.action=='wait-single':
        run=args.output/f'{ds}/seed-2024/B3'
        count=len(protocol['datasets'][ds]['categories'])
        print(f'WAIT {scenario(protocol,ds)} B3 only: original TOF/detector start after all {count} existing responses; zero new Codex calls',flush=True)
        while not all((run/f'codex/call-{i:02d}/response.txt').exists() for i in range(count)):
            if (run/'generation-halt.json').exists():
                single_report(args,protocol)
                raise RuntimeError('Generation halted; preserve logs and do not train on incomplete responses')
            time.sleep(5)
    if args.action in ('evaluate-single','wait-single'):
        evaluate_run(args,protocol,ds,2024,'B3')
        single_report(args,protocol)


if __name__=='__main__':main()
