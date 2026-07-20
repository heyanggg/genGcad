import argparse
import pickle
import json
from pathlib import Path
from codex_backend import CodexClient
from gcad import extract_directional_relationships
from dictionary import dayofweek_dict, hour_dict, fr_devices_dict, fr_actions, sp_devices_dict, sp_actions, us_devices_dict, us_actions
from split import Split
from dayse import Dayse
from transtext import Transtext
from transnumber import Transnum
from sppc import SPPC_select, similarity_select
from baseline1 import Anomaly_detection
from baseline2 import Train
from text_translation_matrix import ATM
from extract import Extract
from find_categories import Find_categories
from security_check import security_check

vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
device_dic = {"us": us_devices_dict, "fr": fr_devices_dict, "sp": sp_devices_dict}
act_dic = {"us": us_actions, "fr": fr_actions, "sp": sp_actions}


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
    parser = argparse.ArgumentParser('LLM generation', add_help=False)
    parser.add_argument('--model', default='gpt-5.6-sol', type=str,
                        help='The Codex model used for generation')
    parser.add_argument('--dataset', default='fr', choices=['fr', 'sp', 'us'],
                        help='Name of dataset to train: fr/us/sp')
    parser.add_argument('--ori_env', default='winter', type=str,
                        help='The original home environment: winter/daytime')
    parser.add_argument('--new_env', default='spring', type=str,
                        help='The new home environment: spring/night')
    parser.add_argument('--method', default='SPPC', type=str,
                        help='The compression method: SPPC/similarity/instance')
    parser.add_argument('--threshold', default=0.918, type=float,
                        help="The compression threshold")
    parser.add_argument('--percentage', default=95.5, type=float,
                        help='The anomaly detection threshold percentage')
    parser.add_argument('--need_test', default=True, type=parse_bool,
                        help='The experimental setup: True/False')
    parser.add_argument('--need_generate', default=False, type=parse_bool,
                        help='The experimental setup: True/False')
    parser.add_argument('--codex-bin', default='codex', type=str,
                        help='Path or command name for the authenticated Codex CLI')
    parser.add_argument('--codex-timeout', default=900, type=int,
                        help='Maximum seconds allowed for each Codex generation')
    parser.add_argument('--codex-reasoning-effort', default='medium', type=str,
                        choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'],
                        help='GPT-5.6 reasoning effort used by Codex')
    parser.add_argument('--gcad-history', default=4, type=int,
                        help='Number of previous action positions used by GCAD')
    parser.add_argument('--gcad-epochs', default=50, type=int,
                        help='Maximum GCAD predictor epochs; validation early stopping is enabled')
    parser.add_argument('--gcad-seeds', default=[2024, 2025, 2026], nargs='+', type=int,
                        help='Random seeds used to retain only stable GCAD relationships')
    parser.add_argument('--gcad-output', default=None, type=str,
                        help='GCAD relationship JSON path; defaults to the original environment directory')
    parser.add_argument('--gcad-force', action='store_true',
                        help='Ignore a matching frozen GCAD artifact and retrain')
    return parser


def LLM_call(codex_client, prompt):
    response = codex_client.generate(prompt)
    print(response)
    return response


def build_prompt(device_control_dict, sentence, user_sequence, action_transition, gcad_relationships):
    gcad_guidance = {
        "status": gcad_relationships.get("status", "unknown"),
        "lag_unit": gcad_relationships.get("lag_unit", "subsequent_behavior_positions"),
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
    return "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. " \
           "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences." \
           "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices." \
           f"The set of the possible device and device states: {device_control_dict}" \
           f"{sentence} The user's compressed original sequences of behavior: {user_sequence}. " \
           f"{historical_guidance}" \
           "Your task: First, select the possible new device states from the set of devices and device states which are also possible new user behaviors. " \
           "The second step is to reasonably add possible new user behaviors to the original user behavior sequences. The third step is to reasonably continue and expand the sequence based on user behavior habits." \
           "Requirements:" \
           "1.Please consider the devices that will be used in the new environment as widely as possible based on the set of devices." \
           "2.Please strictly follow the correspondence between the devices and device states to generate. Do not generate device states that do not match the device." \
           "3.Please add as many new devices and device behaviors as possible to better adapt to changes in the environment." \
           "4.Please make sure that the generated sequence is not a single behavior, but a sequence of consecutive behaviors." \
           "5.Please also generate reasonable behavior time when generating, not just a single behavior." \
           "6.The final generated behavior sequences set is in the format of <seq [['...'], ['...'], ['...']] seq>. For example, the sequences set can be like <seq [['Sunday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(21~24)', 'Camera', 'Camera:notification', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning'], ['Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close']] seq>" \
           "Note that each [...] subsequence represents the user's behavior over a period of time. There is no direct correlation between subsequences. At the same time, the final sequence is strictly generated in the format of <seq [['......'], ['......'], ['......']] seq> without line breaks or inconsistent formats." \
           "Please think step by step, and return the final generated user behavior sequence set."

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)

    if args.need_generate:
        Split(args.dataset, args.ori_env, 1)
        gcad_output = args.gcad_output or f'IoT_data/{args.dataset}/{args.ori_env}/gcad_hints.json'
        gcad_relationships = extract_directional_relationships(
            f'IoT_data/{args.dataset}/{args.ori_env}/split_trn.pkl',
            args.dataset,
            gcad_output,
            history=args.gcad_history,
            epochs=args.gcad_epochs,
            seeds=args.gcad_seeds,
            force=args.gcad_force,
        )
        Dayse(args.dataset, args.ori_env)
        if args.method == 'SPPC':
            Path('IoT_model').mkdir(parents=True, exist_ok=True)
            Train(args.dataset, args.ori_env, vocab_dic[args.dataset])
            SPPC_select(args.dataset, args.ori_env, vocab_dic[args.dataset], args.threshold)
        elif args.method == 'similarity':
            similarity_select(args.dataset, args.ori_env, args.threshold)

        all_categories = Find_categories(args.dataset, args.ori_env, args.method, args.threshold)
        device_dict = device_dic[args.dataset]
        actions = act_dic[args.dataset]
        dictionaries = [dayofweek_dict, hour_dict, device_dict, actions]

        ATM(args.dataset, args.ori_env, actions)
        Transtext(args.dataset, args.ori_env, args.threshold, args.method, all_categories, dictionaries)

        if args.new_env == 'spring':
            sentence = f'The previous environment is {args.ori_env}. The changed environment is warm {args.new_env}.'
        elif args.new_env == 'night':
            sentence = f'The previous environment: user is active during the {args.ori_env} and rest at {args.new_env}. The changed environment: user is active at {args.new_env} and rest during the {args.ori_env}.'
        elif args.new_env == 'multiple':
            sentence = f'The previous environment was for a {args.ori_env} person to be at home, and the changed environment is for {args.new_env} people to be at home'

        with open(f'{args.dataset}_keys_best.txt', 'r') as file:
            device_control_dict = file.read()

        with open(f'IoT_data/{args.dataset}/{args.ori_env}/action_transitions.json', 'r', encoding='utf-8') as f:
            action_transition = json.load(f)
        codex_client = CodexClient(
            model=args.model,
            executable=args.codex_bin,
            timeout=args.codex_timeout,
            reasoning_effort=args.codex_reasoning_effort,
        )

        for day in all_categories:
            with open(f'IoT_data/{args.dataset}/{args.ori_env}/trn_day_{day}_{args.method}_th={args.threshold}_text.pkl', 'rb') as file3:
                user_sequence = pickle.load(file3)
                print(len(user_sequence))
            prompt = build_prompt(
                device_control_dict,
                sentence,
                user_sequence,
                action_transition,
                gcad_relationships,
            )
            response = LLM_call(codex_client, prompt)
            with open(f'IoT_data/{args.dataset}/{args.new_env}/{args.dataset}_{args.new_env}_generation_day_{day}_{args.method}_th={args.threshold}_{args.model}.pkl', 'wb') as f3:
                pickle.dump(response, f3)

        Extract(args.dataset, args.new_env, args.threshold, args.method, args.model, all_categories)
        Transnum(args.dataset, args.new_env, args.threshold, args.method, args.model, all_categories, dictionaries)
        Path('check_model').mkdir(parents=True, exist_ok=True)
        security_check(args.dataset, args.new_env, args.threshold, args.method, args.model)


    if args.need_test:
        Anomaly_detection(args.dataset, args.new_env, args.threshold, args.method, args.model, args.percentage)
