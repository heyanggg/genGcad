import argparse
import hashlib
import json
import pickle
from pathlib import Path

from baseline1 import Anomaly_detection
from baseline2 import Train
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
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--gcad-history", "--gcad_history", default=4, type=int)
    parser.add_argument("--gcad-epochs", "--gcad_epochs", default=50, type=int)
    parser.add_argument(
        "--gcad-seeds", "--gcad_seeds", default=[2024, 2025, 2026], nargs="+", type=int
    )
    parser.add_argument("--gcad-output", "--gcad_output", default=None)
    parser.add_argument("--gcad-force", action="store_true")
    return parser


def build_prompt(
    device_control_dict,
    sentence,
    user_sequence,
    action_transition,
    gcad_relationships,
):
    gcad_guidance = {
        "status": gcad_relationships.get("status", "unknown"),
        "lag_unit": gcad_relationships.get(
            "lag_unit", "subsequent_behavior_positions"
        ),
        "directional_relationships": gcad_relationships.get(
            "lagged_behavior_relations", []
        ),
    }
    historical_guidance = (
        "Historical behavior guidance contains two kinds of patterns: "
        "1. Immediate transition patterns: These describe actions that frequently occur directly "
        "after another action. "
        f"{json.dumps(action_transition, ensure_ascii=False)} "
        "2. Lagged directional dependency patterns (GCAD-derived): These are validated predictive "
        "dependencies, not proof of real-world causality. A source action may help predict a "
        "target action after the indicated number of subsequent behavior positions. "
        f"{json.dumps(gcad_guidance, ensure_ascii=False)} "
        "Use both kinds of patterns as soft guidance. Immediate transition patterns mainly guide "
        "adjacent actions. Lagged dependency patterns may be satisfied after intermediate actions. "
        "Do not force every pattern to appear in every sequence. Adapt the behaviors when they "
        "conflict with the new environment. "
    )
    return (
        "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. "
        "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences."
        "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices."
        f"The set of the possible device and device states: {device_control_dict}"
        f"{sentence} The user's compressed original sequences of behavior: {user_sequence}. "
        f"{historical_guidance}"
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
        codex_reasoning_effort=args.codex_reasoning_effort,
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
        gcad_output = Path(args.gcad_output) if args.gcad_output else (
            Path("artifacts") / "gcad" / args.dataset / args.ori_env / "gcad_hints.json"
        )
        gcad_relationships = extract_directional_relationships(
            Path("IoT_data") / args.dataset / args.ori_env / "split_trn.pkl",
            args.dataset,
            gcad_output,
            history=args.gcad_history,
            epochs=args.gcad_epochs,
            seeds=args.gcad_seeds,
            split_seed=args.experiment_seed,
            force=args.gcad_force,
        )
        run.update(
            status="compressing",
            gcad={
                "status": gcad_relationships.get("status"),
                "artifact_fingerprint": gcad_relationships.get("artifact_fingerprint"),
                "relationship_count": len(
                    gcad_relationships.get("lagged_behavior_relations", [])
                ),
                "path": str(gcad_output),
            },
        )

        Dayse(args.dataset, args.ori_env)
        if args.method == "SPPC":
            Train(
                args.dataset,
                args.ori_env,
                VOCABULARY_SIZE[args.dataset],
                seed=args.experiment_seed,
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
                gcad_relationships,
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
        tof_result = security_check(
            args.dataset,
            args.new_env,
            args.threshold,
            args.method,
            artifact_model,
            seed=args.experiment_seed,
            keep_intermediates=args.keep_intermediates,
        )
        run.update(
            status="completed",
            outputs={
                "generated_numeric": str(
                    filter_dir
                    / f"{args.dataset}_{args.new_env}_generation_{args.method}_th={args.threshold}_{artifact_model}_seq.pkl"
                ),
                "tof_final": tof_result["output_path"],
                "tof_final_count": tof_result["final_count"],
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
    if args.need_generate:
        manifest = run_generation(args, config)
        print(f"Generation completed: {manifest['outputs']}")
    if args.need_test:
        result = run_anomaly_detection(args, config)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
