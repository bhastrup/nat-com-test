import itertools, sys, json, glob
from typing import Callable, List, Dict

import submitit

from src.tools.arg_parser_pretrain import build_default_argparser_pretrain
from src.tools.arg_parser import build_default_argparser


def load_default_configuration(
        use_old_config: bool = False,
        load_model: str = None,
        build_default_argparser_pretrain: Callable = build_default_argparser_pretrain,
        build_default_argparser: Callable = build_default_argparser,    
) -> dict:
    if use_old_config and load_model:
        config = load_old_config(load_model)
    else:
        config = get_config_pretrain(build_default_argparser_pretrain)
        config["config_ft"] = get_config(build_default_argparser)
    
    return config


def load_old_config(model_path: str = None) -> dict:
    """ Loads the config from the log folder. Assumes that logs and models are located next to each other"""
    model_dir = '/'.join(model_path.split('/')[:-1])
    # run_dir = '/'.join(model_dir.split('/')[:-1])
    logs_dir = model_dir.replace('models', 'logs')
    results_dir = model_dir.replace('models', 'results')
    data_dir = model_dir.replace('models', 'data')

    config_files = glob.glob(f'{logs_dir}/*.json')
    if not config_files:
        raise FileNotFoundError(f"No JSON configuration file found in {logs_dir}")

    with open(config_files[0]) as f:
        config = json.load(f)
    
    config['model_dir'] = model_dir
    config['log_dir'] = logs_dir
    config['results_dir'] = results_dir
    config['data_dir'] = data_dir

    return config


    

def get_sublog_name(EXPERIMENT_NAME: str, dt_string: str) -> str:
    return f"sublogs/" + EXPERIMENT_NAME + '_' + dt_string + '_main'


def get_config(build_default_argparser: Callable = build_default_argparser) -> dict:
    parser = build_default_argparser()
    args = parser.parse_args()
    config = vars(args)
    return config


def get_config_pretrain(build_default_argparser_pretrain: Callable = build_default_argparser_pretrain) -> dict:
    parser = build_default_argparser_pretrain()
    args = parser.parse_args()
    config = vars(args)
    return config


def submit_jobs(
    submit_fn: Callable,
    executor: submitit.AutoExecutor,
    parameter_dicts: List[Dict],
    ask_permission: bool = True,
    use_submitit: bool = True
):
    """Sanity check and submit job array"""
    print(f'parameter_dicts: {parameter_dicts}')
    # check_for_duplicates(parameter_dicts)

    response = input(f"Submit {len(parameter_dicts)} jobs? (y/n) ") if ask_permission else "y"
    if response.lower() == "y":
        # parameter_dicts = [{key: value for key, value in parameter_dict.items() if value is not None} \
        #                     for parameter_dict in parameter_dicts]
        if use_submitit:
            jobs = executor.map_array(submit_fn, parameter_dicts)
            for job in jobs:
                print(f"Submitted job: {job.job_id}")
        else:
            for parameter_dict in parameter_dicts:
                param_dict = parameter_dict.copy()
                # param_dict.update({'save_to_wandb': False})
                submit_fn(param_dict)
    else:
        print("Aborting.")


def check_for_duplicates(parameter_dicts):
    """Check that all parameter combinations are unique"""

    # Convert each dictionary to a sorted tuple of its items,
    # with nested dictionaries converted to string representations
    parameter_tuples = [
        tuple((k, json.dumps(v, sort_keys=True)) for k, v in sorted(d.items()))
        for d in parameter_dicts
    ]
    assert len(parameter_dicts) == len(set(parameter_tuples)), "Duplicate parameter combinations"


def generate_parameter_combinations(config, sweep_params, exp_name, ):
    """
    Generate parameter combinations by taking the Cartesian product of the values in sweep_params.
    Combine each parameter combination with the non-sweep parameters and the config dictionary.

    Args:
        config: A dictionary of configuration options.
        sweep_params: A dictionary of sweep parameters and their possible values.
        non_sweep_params: A dictionary of non-sweep parameters and their values.

    Returns:
        A list of dictionaries, where each dictionary contains a set of parameter values.
    """
    parameter_dicts = []
    names = []
    sweep_combinations = itertools.product(*sweep_params.values())
    for sweep_combination in sweep_combinations:
            name = "-".join([f"{key}_{value}" for key, value in zip(sweep_params.keys(), sweep_combination)])
            names.append(name)
            parameter_dict = config.copy()
            parameter_dict.update(dict(zip(sweep_params.keys(), sweep_combination)))
            parameter_dicts.append(parameter_dict)

    for parameter_dict, dict_name in zip(parameter_dicts, names):
        parameter_dict = set_directories(parameter_dict, exp_name, dict_name)
        parameter_dict.update(
            dict(name=parameter_dict['model'],
                 wandb_name=dict_name)
        )

    return parameter_dicts


def set_directories(parameter_dict: dict, exp_name: str, sweep_name: str):
    parameter_dict['log_dir']=f"pretrain_runs/{exp_name}/{sweep_name}/logs"
    parameter_dict['model_dir']=f"pretrain_runs/{exp_name}/{sweep_name}/models"
    parameter_dict['data_dir']=f"pretrain_runs/{exp_name}/{sweep_name}/data"
    parameter_dict['results_dir']=f"pretrain_runs/{exp_name}/{sweep_name}/results"
    return parameter_dict



def set_default_resources(executor):
    if executor._executor.__class__.__name__ == 'SlurmExecutor':
        executor._command = "/home/energy/bjaha/miniconda3/envs/delight-rl/bin/python"
        executor.update_parameters(
            slurm_partition="sm3090",
            # gres="gpu:RTX3090:1",
            slurm_num_gpus=1,
            cpus_per_task=8,
            tasks_per_node=1, # the new one
            slurm_nodes=1,
            slurm_time="0-03:00:00",
            # slurm_mail_type="END,FAIL",
            mem_gb=16
        )
    elif executor._executor.__class__.__name__ == 'LocalExecutor':
        executor.update_parameters(
            timeout_min=60*3,
            gpus_per_node=0,
            cpus_per_task=2,
            nodes=1,
            tasks_per_node=1,
            mem_gb=4,
        )
    else:
        print(f"Unknown executor type: {executor._executor.__class__.__name__}")
        sys.exit(1)

    return executor