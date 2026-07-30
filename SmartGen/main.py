import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

from baseline1 import Anomaly_detection
from baseline2 import Train
from archiving import ARCHIVE_STATUSES, archive_completed_experiment
from codex_backend import CodexClient
from dayse import Dayse
from dictionary import (
    dayofweek_dict,
    fr_actions,
    fr_devices_dict,
    hour_dict,
    sp_actions,
    sp_devices_dict,
    us_actions,
    us_devices_dict,
)
from experiment import (
    ExperimentConfig,
    ExperimentRun,
    atomic_pickle_dump,
    atomic_write_text,
    enter_smartgen_root,
)
from extract import Extract
from find_categories import Find_categories
from gcad import extract_directional_relationships
from security_check import security_check
from split import Split
from sppc import SPPC_select, similarity_select
from text_translation_matrix import ATM
from transnumber import Transnum
from transtext import Transtext


VOCABULARY_SIZE = {"fr": 223, "us": 269, "sp": 235}
DEVICE_DICTIONARIES = {
    "us": us_devices_dict,
    "fr": fr_devices_dict,
    "sp": sp_devices_dict,
}
ACTION_DICTIONARIES = {"us": us_actions, "fr": fr_actions, "sp": sp_actions}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def get_args_parser():
    parser = argparse.ArgumentParser("SmartGen + GCAD experiment runner")
    parser.add_argument("--model", default="gpt-5.6-sol", help="Codex generation model")
    parser.add_argument("--run-id", "--run_id", default="run1", help="Unique generation replicate label")
    parser.add_argument("--experiment-seed", "--experiment_seed", default=2024, type=int)
    parser.add_argument("--dataset", default="fr", choices=["fr", "sp", "us"])
    parser.add_argument(
        "--ori-env", "--ori_env", default="winter", choices=["winter", "daytime", "single"]
    )
    parser.add_argument(
        "--new-env", "--new_env", default="spring", choices=["spring", "night", "multiple"]
    )
    parser.add_argument(
        "--method", default="SPPC", choices=["SPPC", "similarity", "instance"]
    )
    parser.add_argument("--threshold", default=0.918, type=float)
    parser.add_argument("--percentage", default=95.5, type=float)
    parser.add_argument("--need-test", "--need_test", default=False, type=parse_bool)
    parser.add_argument("--need-generate", "--need_generate", default=False, type=parse_bool)
    parser.add_argument(
        "--reuse-sppc-selection-dir",
        default=None,
        help=(
            "Directory containing archived trn_day_<0-6>_SPPC selection files to "
            "reuse verbatim; source paths and SHA-256 values are recorded in the manifest"
        ),
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep TOF candidate split files after final filtering",
    )
    parser.add_argument("--codex-bin", "--codex_bin", default="codex")
    parser.add_argument("--codex-timeout", "--codex_timeout", default=900, type=int)
    parser.add_argument(
        "--codex-reasoning-effort", "--codex_reasoning_effort",
        default="none",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument(
        "--prompt-profile", "--prompt_profile",
        default="environment-aware",
        choices=["original", "environment-aware"],
        help=(
            "Use the byte-equivalent upstream prompt or add explicit environment and "
            "sequence-shape constraints for GPT-5.6"
        ),
    )
    parser.add_argument("--gcad-history", "--gcad_history", default=4, type=int)
    parser.add_argument("--gcad-epochs", "--gcad_epochs", default=50, type=int)
    parser.add_argument(
        "--gcad-seeds", "--gcad_seeds", default=[2024, 2025, 2026], nargs="+", type=int
    )
    parser.add_argument("--gcad-output", "--gcad_output", default=None)
    parser.add_argument("--gcad-force", action="store_true")
    parser.add_argument(
        "--gcad-mode", "--gcad_mode",
        default="auto",
        choices=["auto", "off", "require"],
        help=(
            "auto uses stable GCAD guidance when available; off suppresses it; "
            "require fails unless stable relationships are available"
        ),
    )
    parser.add_argument(
        "--gcad-prompt-max-relationships", "--gcad_prompt_max_relationships",
        default=12,
        type=int,
        help="Maximum number of balanced GCAD relationships included in a generation prompt",
    )
    parser.add_argument(
        "--gcad-prompt-max-per-target", "--gcad_prompt_max_per_target",
        default=4,
        type=int,
        help="Maximum prompt relationships sharing the same GCAD target action",
    )
    parser.add_argument(
        "--archive-status",
        default="completed",
        choices=sorted(ARCHIVE_STATUSES),
        help="Archive a complete generation+detection run under this status",
    )
    parser.add_argument(
        "--no-archive",
        action="store_false",
        dest="archive_run",
        help="Do not create the automatic self-contained archive",
    )
    parser.set_defaults(archive_run=True)
    return parser


def compose_smartgen_prompt(
    device_control_dict,
    sentence,
    user_sequence,
    action_transition,
    additional_guidance="",
):
    return (
        "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. "
        "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences."
        "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices."
        f"The set of the possible device and device states: {device_control_dict}"
        f"{sentence} The user's compressed original sequences of behavior: {user_sequence}. User's behavior habits: {action_transition}"
        f"{additional_guidance}"
        "Your task: First, select the possible new device states from the set of devices and device states which are also possible new user behaviors. "
        "The second step is to reasonably add possible new user behaviors to the original user behavior sequences. The third step is to reasonably continue and expand the sequence based on user behavior habits."
        "Requirements:"
        "1.Please consider the devices that will be used in the new environment as widely as possible based on the set of devices."
        "2.Please strictly follow the correspondence between the devices and device states to generate. Do not generate device states that do not match the device."
        "3.Please add as many new devices and device behaviors as possible to better adapt to changes in the environment."
        "4.Please make sure that the generated sequence is not a single behavior, but a sequence of consecutive behaviors."
        "5.Please also generate reasonable behavior time when generating, not just a single behavior."
        "6.The final generated behavior sequences set is in the format of <seq [['...'], ['...'], ['...']] seq>. For example, the sequences set can be like <seq [['Sunday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(21~24)', 'Camera', 'Camera:notification', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning'], ['Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close']] seq>"
        "Note that each [...] subsequence represents the user's behavior over a period of time. There is no direct correlation between subsequences. At the same time, the final sequence is strictly generated in the format of <seq [['......'], ['......'], ['......']] seq> without line breaks or inconsistent formats."
        "Please think step by step, and return the final generated user behavior sequence set."
    )


def build_prompt(
    device_control_dict,
    sentence,
    user_sequence,
    action_transition,
    gcad_relationships,
    target_environment=None,
    prompt_profile="original",
):
    guidance_parts = []
    if prompt_profile == "environment-aware":
        guidance_parts.append(environment_generation_guidance(target_environment))
    elif prompt_profile != "original":
        raise ValueError(f"unsupported prompt profile: {prompt_profile!r}")

    relationships = gcad_relationships.get("lagged_behavior_relations", [])
    if gcad_relationships.get("status") == "ready" and relationships:
        gcad_guidance = {
            "lag_unit": gcad_relationships.get(
                "lag_unit", "subsequent_behavior_positions"
            ),
            "directional_relationships": relationships,
        }
        guidance_parts.append(
            "Directional behavior relationship guidance: "
            "The following relationships were extracted from historical normal behavior "
            "sequences. When a source behavior occurs, the corresponding target behavior is "
            "often influenced within the indicated number of subsequent behavior positions. "
            "Use these relationships as soft guidance. Preserve them when they are compatible "
            "with the new environmental context, but do not force every relationship to appear "
            "in every generated sequence. "
            f"{json.dumps(gcad_guidance, ensure_ascii=False)}"
        )
    additional_guidance = "".join(f" {item} " for item in guidance_parts if item)
    return compose_smartgen_prompt(
        device_control_dict,
        sentence,
        user_sequence,
        action_transition,
        additional_guidance,
    )


def select_gcad_guidance(gcad_relationships, mode):
    relationships = gcad_relationships.get("lagged_behavior_relations", [])
    ready = gcad_relationships.get("status") == "ready" and bool(relationships)
    if mode == "auto":
        return gcad_relationships
    if mode == "off":
        return {
            "status": "disabled",
            "disabled_reason": "disabled_by_experiment_configuration",
            "lagged_behavior_relations": [],
        }
    if mode == "require":
        if not ready:
            reason = gcad_relationships.get(
                "disabled_reason", "no_stable_directional_relationships"
            )
            raise RuntimeError(
                f"GCAD guidance was required but is unavailable: {reason}"
            )
        return gcad_relationships
    raise ValueError(f"unsupported GCAD mode: {mode!r}")


def prepare_gcad_stage(args):
    """Run GCAD only for GCAD-enabled experiments.

    A GCAD-off run is the original SmartGen pipeline, so it must neither
    extract/read a GCAD artifact nor reuse an SSC selection from an on run.
    """
    if args.gcad_mode == "off":
        if getattr(args, "reuse_sppc_selection_dir", None):
            raise ValueError(
                "--gcad-mode off cannot be combined with "
                "--reuse-sppc-selection-dir; a GCAD-off baseline must execute "
                "the original SmartGen SSC stage independently"
            )
        return (
            {
                "status": "skipped",
                "disabled_reason": "gcad_module_not_executed_for_baseline",
                "lagged_behavior_relations": [],
            },
            None,
            False,
        )

    gcad_output = Path(args.gcad_output) if args.gcad_output else (
        Path("artifacts") / "gcad" / args.dataset / args.ori_env / "gcad_hints.json"
    )
    relationships = extract_directional_relationships(
        Path("IoT_data") / args.dataset / args.ori_env / "split_trn.pkl",
        args.dataset,
        gcad_output,
        history=args.gcad_history,
        epochs=args.gcad_epochs,
        seeds=args.gcad_seeds,
        split_seed=args.experiment_seed,
        force=args.gcad_force,
    )
    return relationships, gcad_output, True


def limit_prompt_gcad_relationships(
    gcad_relationships,
    max_relationships=12,
    max_per_target=4,
):
    """Bound prompt dosage while retaining the complete archived GCAD graph."""
    if max_relationships <= 0 or max_per_target <= 0:
        raise ValueError("GCAD prompt relationship limits must be positive")
    relationships = gcad_relationships.get("lagged_behavior_relations", [])
    selected = []
    target_counts = {}
    for relationship in relationships:
        target = relationship.get("target_action")
        if target_counts.get(target, 0) >= max_per_target:
            continue
        selected.append(relationship)
        target_counts[target] = target_counts.get(target, 0) + 1
        if len(selected) >= max_relationships:
            break
    return {
        **gcad_relationships,
        "lagged_behavior_relations": selected,
        "prompt_selection": {
            "artifact_relationship_count": len(relationships),
            "prompt_relationship_count": len(selected),
            "max_relationships": max_relationships,
            "max_per_target": max_per_target,
        },
    }


def environment_generation_guidance(target_environment):
    common = (
        "Generation calibration guidance: Generate coherent consecutive behavior chains rather "
        "than copying singleton compressed representatives as singleton outputs. Do not add "
        "unrelated device actions merely to increase device coverage. Environmental consistency "
        "takes priority over device coverage. "
    )
    if target_environment == "night":
        return common + (
            "For this category, generate 3 to 4 distinct subsequences, normally containing 5 to "
            "8 behavior quadruples each. "
            "For the changed night-active environment, place the main active behaviors in the "
            "available time intervals (18~21), (21~24), (0~3), and (3~6). The daytime intervals "
            "should remain a small minority rather than disappearing completely: across the "
            "complete response, normally place about 5 percent of plausible transitional "
            "exceptions in (6~9) or (9~12). The later daytime intervals (12~15) and (15~18) "
            "should remain rare and be used only when a behavior genuinely requires them."
        )
    if target_environment == "spring":
        return common + (
            "For this category, generate distinct subsequences normally containing 4 to 6 "
            "behavior quadruples each. Preserve the target-domain sequence-length distribution "
            "and do not lengthen a chain merely to include more directional relationships. "
            "For the changed warm spring environment, prefer behavior changes that are plausible "
            "for warmer weather and avoid retaining winter-specific heating behavior unless the "
            "individual sequence provides a clear reason."
        )
    if target_environment == "multiple":
        return common + (
            "For the changed multiple-resident environment, represent plausible variation or "
            "overlap between household routines while keeping each subsequence internally coherent."
        )
    raise ValueError(f"unsupported target environment: {target_environment!r}")


def summarize_environment_adherence(sequences, target_environment):
    """Report prompt adherence without filtering or using anomaly labels."""
    hour_counts = {str(index): 0 for index in range(len(hour_dict))}
    behavior_count = 0
    for sequence in sequences:
        for index in range(1, len(sequence), 4):
            hour = int(sequence[index])
            hour_counts[str(hour)] = hour_counts.get(str(hour), 0) + 1
            behavior_count += 1
    summary = {
        "target_environment": target_environment,
        "behavior_count": behavior_count,
        "hour_counts": hour_counts,
        "used_for_filtering": False,
    }
    if target_environment == "night":
        night_count = sum(hour_counts[str(index)] for index in (0, 1, 6, 7))
        summary["preferred_hour_codes"] = [0, 1, 6, 7]
        summary["preferred_behavior_count"] = night_count
        summary["preferred_behavior_ratio"] = (
            night_count / behavior_count if behavior_count else 0.0
        )
    return summary


def should_preserve_tof_intermediates(args):
    return bool(args.keep_intermediates or args.archive_run)


def environment_sentence(original_environment, target_environment):
    if target_environment == "spring":
        return (
            f"The previous environment is {original_environment}. "
            f"The changed environment is warm {target_environment}."
        )
    if target_environment == "night":
        return (
            f"The previous environment: user is active during the {original_environment} and rest "
            f"at {target_environment}. The changed environment: user is active at "
            f"{target_environment} and rest during the {original_environment}."
        )
    return (
        f"The previous environment was for a {original_environment} person to be at home, "
        f"and the changed environment is for {target_environment} people to be at home"
    )


def experiment_config_from_args(args):
    return ExperimentConfig(
        dataset=args.dataset,
        original_environment=args.ori_env,
        target_environment=args.new_env,
        method=args.method,
        threshold=args.threshold,
        model=args.model,
        run_id=args.run_id,
        experiment_seed=args.experiment_seed,
        gcad_seeds=tuple(args.gcad_seeds),
        gcad_history=args.gcad_history,
        gcad_epochs=args.gcad_epochs,
        gcad_mode=args.gcad_mode,
        gcad_prompt_max_relationships=args.gcad_prompt_max_relationships,
        gcad_prompt_max_per_target=args.gcad_prompt_max_per_target,
        codex_reasoning_effort=args.codex_reasoning_effort,
        prompt_profile=args.prompt_profile,
    )


def load_resumable_response(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            response = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    return response if CodexClient.is_valid_response(response) else None


def run_generation(args, config):
    run = ExperimentRun(config)
    run.update(status="preparing")
    artifact_model = config.artifact_model
    target_dir = Path("IoT_data") / args.dataset / args.new_env
    filter_dir = Path("filter_data") / args.dataset / args.new_env
    target_dir.mkdir(parents=True, exist_ok=True)
    filter_dir.mkdir(parents=True, exist_ok=True)

    try:
        Split(args.dataset, args.ori_env, 1)
        gcad_relationships, gcad_output, gcad_module_executed = (
            prepare_gcad_stage(args)
        )
        prompt_gcad_relationships = select_gcad_guidance(
            gcad_relationships, args.gcad_mode
        )
        prompt_gcad_relationships = limit_prompt_gcad_relationships(
            prompt_gcad_relationships,
            max_relationships=args.gcad_prompt_max_relationships,
            max_per_target=args.gcad_prompt_max_per_target,
        )
        prompt_selection = prompt_gcad_relationships["prompt_selection"]
        run.update(
            status="compressing",
            pipeline_mode=(
                "original_smartgen_without_gcad"
                if args.gcad_mode == "off"
                else "smartgen_with_gcad_stage"
            ),
            gcad={
                "status": gcad_relationships.get("status"),
                "disabled_reason": gcad_relationships.get("disabled_reason"),
                "artifact_fingerprint": gcad_relationships.get("artifact_fingerprint"),
                "relationship_count": len(
                    gcad_relationships.get("lagged_behavior_relations", [])
                ),
                "artifact_relationship_count": prompt_selection[
                    "artifact_relationship_count"
                ],
                "prompt_relationship_count": prompt_selection[
                    "prompt_relationship_count"
                ],
                "prompt_relationship_limits": {
                    "max_relationships": prompt_selection["max_relationships"],
                    "max_per_target": prompt_selection["max_per_target"],
                },
                "path": str(gcad_output) if gcad_output is not None else None,
                "mode": args.gcad_mode,
                "module_executed": gcad_module_executed,
                "guidance_enabled": (
                    prompt_gcad_relationships.get("status") == "ready"
                    and bool(
                        prompt_gcad_relationships.get(
                            "lagged_behavior_relations", []
                        )
                    )
                ),
            },
        )

        Dayse(args.dataset, args.ori_env)
        if args.method == "SPPC":
            if args.reuse_sppc_selection_dir:
                selection_dir = Path(args.reuse_sppc_selection_dir).resolve()
                selection_files = {}
                for day in range(7):
                    filename = (
                        f"trn_day_{day}_SPPC_th={args.threshold}.pkl"
                    )
                    source = selection_dir / filename
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"cannot reuse missing SPPC selection: {source}"
                        )
                    destination = (
                        Path("IoT_data") / args.dataset / args.ori_env / filename
                    )
                    shutil.copy2(source, destination)
                    selection_files[filename] = hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest()
                run.update(
                    compression={
                        "method": "SPPC",
                        "training_mode": "reused_archived_selection",
                        "selection_source_dir": str(selection_dir),
                        "selection_sha256": selection_files,
                    }
                )
            else:
                Train(
                    args.dataset,
                    args.ori_env,
                    VOCABULARY_SIZE[args.dataset],
                    seed=args.experiment_seed,
                )
                model_path = Path("IoT_model") / (
                    f"Transformer_{args.dataset}_{args.ori_env}_15epoch.pth"
                )
                run.update(
                    compression={
                        "method": "SPPC",
                        "training_mode": "trained",
                        "checkpoint_path": str(model_path),
                        "checkpoint_sha256": hashlib.sha256(
                            model_path.read_bytes()
                        ).hexdigest(),
                    }
                )
                SPPC_select(
                    args.dataset,
                    args.ori_env,
                    VOCABULARY_SIZE[args.dataset],
                    args.threshold,
                    seed=args.experiment_seed,
                )
        elif args.method == "similarity":
            similarity_select(args.dataset, args.ori_env, args.threshold)

        all_categories = Find_categories(
            args.dataset, args.ori_env, args.method, args.threshold
        )
        device_dictionary = DEVICE_DICTIONARIES[args.dataset]
        actions = ACTION_DICTIONARIES[args.dataset]
        dictionaries = [dayofweek_dict, hour_dict, device_dictionary, actions]
        ATM(args.dataset, args.ori_env, actions)
        Transtext(
            args.dataset,
            args.ori_env,
            args.threshold,
            args.method,
            all_categories,
            dictionaries,
        )

        sentence = environment_sentence(args.ori_env, args.new_env)
        device_control_dict = Path(f"{args.dataset}_keys_best.txt").read_text(
            encoding="utf-8"
        )
        action_transition = json.loads(
            (Path("IoT_data") / args.dataset / args.ori_env / "action_transitions.json").read_text(
                encoding="utf-8"
            )
        )
        codex_client = None
        run.update(status="generating", categories=[str(item) for item in all_categories])
        for category in all_categories:
            category_name = str(category)
            source_path = (
                Path("IoT_data")
                / args.dataset
                / args.ori_env
                / f"trn_day_{category}_{args.method}_th={args.threshold}_text.pkl"
            )
            with source_path.open("rb") as handle:
                user_sequence = pickle.load(handle)
            prompt = build_prompt(
                device_control_dict,
                sentence,
                user_sequence,
                action_transition,
                prompt_gcad_relationships,
                target_environment=args.new_env,
                prompt_profile=args.prompt_profile,
            )
            atomic_write_text(run.prompt_path(category_name), prompt)
            output_path = target_dir / (
                f"{args.dataset}_{args.new_env}_generation_day_{category}_{args.method}_"
                f"th={args.threshold}_{artifact_model}.pkl"
            )
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            response = (
                load_resumable_response(output_path)
                if args.resume and run.prompt_matches(category_name, prompt)
                else None
            )
            if response is None:
                if codex_client is None:
                    codex_client = CodexClient(
                        model=args.model,
                        executable=args.codex_bin,
                        timeout=args.codex_timeout,
                        reasoning_effort=args.codex_reasoning_effort,
                    )
                    run.update(generation_backend=codex_client.generation_protocol)
                response = codex_client.generate(prompt)
                atomic_pickle_dump(output_path, response)
            atomic_write_text(run.response_path(category_name), response + "\n")
            run.record_category(category_name, output_path, prompt_sha256)

        Extract(
            args.dataset,
            args.new_env,
            args.threshold,
            args.method,
            artifact_model,
            all_categories,
        )
        Transnum(
            args.dataset,
            args.new_env,
            args.threshold,
            args.method,
            artifact_model,
            all_categories,
            dictionaries,
        )
        generated_numeric_path = filter_dir / (
            f"{args.dataset}_{args.new_env}_generation_{args.method}_th={args.threshold}_"
            f"{artifact_model}_seq.pkl"
        )
        with generated_numeric_path.open("rb") as handle:
            generated_numeric = pickle.load(handle)
        adherence = summarize_environment_adherence(
            generated_numeric, args.new_env
        )
        tof_result = security_check(
            args.dataset,
            args.new_env,
            args.threshold,
            args.method,
            artifact_model,
            seed=args.experiment_seed,
            keep_intermediates=should_preserve_tof_intermediates(args),
        )
        run.update(
            status="completed",
            outputs={
                "generated_numeric": str(generated_numeric_path),
                "tof_final": tof_result["output_path"],
                "tof_final_count": tof_result["final_count"],
            },
            environment_adherence=adherence,
            tof={
                "input_count": tof_result["input_count"],
                "first_pass_count": tof_result["first_pass_count"],
                "final_count": tof_result["final_count"],
                "intermediates_preserved": should_preserve_tof_intermediates(args),
            },
        )
        return run.manifest
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def run_anomaly_detection(args, config):
    return Anomaly_detection(
        args.dataset,
        args.new_env,
        args.threshold,
        args.method,
        config.artifact_model,
        args.percentage,
        seed=args.experiment_seed,
    )


def main(argv=None):
    enter_smartgen_root()
    parser = get_args_parser()
    args = parser.parse_args(argv)
    if not args.need_generate and not args.need_test:
        parser.print_help()
        return 0
    config = experiment_config_from_args(args)
    manifest = None
    result = None
    if args.need_generate:
        manifest = run_generation(args, config)
        print(f"Generation completed: {manifest['outputs']}")
    if args.need_test:
        result = run_anomaly_detection(args, config)
        print(json.dumps(result, indent=2))
    if args.archive_run and manifest is not None and result is not None:
        archive_path = archive_completed_experiment(
            config,
            manifest,
            result,
            status=args.archive_status,
        )
        print(f"Experiment archived: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
