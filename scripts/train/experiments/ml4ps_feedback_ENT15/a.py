import sys
from datetime import datetime
import numpy as np
import submitit

from xtb.ase.calculator import XTB

from src.performance.energetics import EnergyUnit
from scripts.train.run_bc import pretrain as submit_fn
from launchers.launch_utils import (
    load_default_configuration, get_sublog_name, generate_parameter_combinations, submit_jobs
)


TRAIN_MODE = 'pretrain'

def main() -> None:
    
    load_model = None # 'pretrain_runs/NIPS-AV-R/relax_steps_final_6/models/pretrain_run-0_CP-6_steps-15000.model'
    load_latest = False
    use_old_config = False


    # load_model = 'pretrain_runs/NIPS-F/name_NIPS-F/models/pretrain_run-0_steps-4300.model'
    config = load_default_configuration(use_old_config, load_model)
    if use_old_config and (load_model or load_latest):
        config['load_model'] = load_model
        config['load_latest'] = load_latest
        config['safe_xtb'] = False
        submit_jobs(
            submit_fn=submit_fn,
            executor=None,
            parameter_dicts=[config],
            ask_permission=False,
            use_submitit=False
        )


    wandb_group = 'final-ent15-A'
    experiment_name = wandb_group

    dt_string = datetime.now().strftime("%d_%m_%H_%M")
    executor = submitit.AutoExecutor(folder=get_sublog_name(experiment_name, dt_string))
    if executor._executor.__class__.__name__ == 'SlurmExecutor':
        executor.update_parameters(
            slurm_partition="sm3090el8", #"a100", # "a100_week", # 
            gpus_per_node=1, #slurm_num_gpus=1,
            tasks_per_node=1, # the new one
            cpus_per_task=8,
            slurm_nodes=1,
            slurm_time="7-00:00:00",
            # mem_gb=64
        )
        jobs_args = dict(
            save_to_wandb=True,
            wandb_mode='online',

            mini_batch_size=256,
            network_width=128,
            num_interactions=3,
            num_epochs=1000,
            learning_rate=2e-5,
            save_freq=10,
            eval_freq=1500,
            eval_freq_fast=100,
        )
        config.update(jobs_args)
    elif executor._executor.__class__.__name__ == 'LocalExecutor':
        executor.update_parameters(
            timeout_min=40, # units of minutes
            gpus_per_node=1,
            cpus_per_task=14,
            nodes=1,
            tasks_per_node=1,
            mem_gb=12,
        )
        jobs_args = dict(
            save_to_wandb=False,
            wandb_mode='online',

            mini_batch_size=128,
            num_steps_per_iter=128,
            network_width=128,
            num_interactions=3,
            num_epochs=50,
            learning_rate=3e-4,
            save_freq=10,
            eval_freq=500,
            eval_freq_fast=50

        )
        config.update(jobs_args)
    
    # Sweep parameter names and values
    sweep_params = dict(
        #relax_steps_final=[4, 8, 12],
        #exp=['AV'],
        #nhf=[config['no_hydrogen_focus']],
        # m=[config['model']]
        seed=[0, 1, 2]
    )

    # Other parameters to be included in all runs
    config.update(
        dict(
            wandb_project='molgym-pretrain',
            train_mode=TRAIN_MODE,
            wandb_job_type='',
            entity='bhastrup',
            #name=experiment_name,
            wandb_group=wandb_group,

            # Data
            mol_dataset='QM7',
            energy_unit='eV',

            # Environment
            relax_steps_final=0,
            reward_coefs = {'rew_atomisation': 1.0} , # , 'rew_valid': 3.0},
            #reward_coefs = {'rew_formation': 1.0}, #, 'rew_valid': 3.0,  'rew_atomisation': 1.0},
            min_reward=-3, # -6
            safe_xtb=False,

            partial_canvas=False, 
            # Only used if partial_canvas:
            # decom_p_random=1., 
            # n_atoms_to_place=2,


            # Agent policy
            model = 'painn', # 'reveal_nn' 'covariant' 'painn_internal_multimodal',
            rms_norm_update = False,


            # Model IO
            device='cuda',
            checkpoints = [int(c) for c in np.arange(0, 51) * int(2500)],
            load_model = load_model,

            # Training scheme
            # Online
            rl_algo_online='PPO',
            # Offline
            pretrain_every_k_iter=None,
            rl_algo_pretrain='bc',
            dataloader_bs=16,
            grad_steps_offline=1,
            lr_ratio=1.0,

        )
    )

    config["config_ft"].update(
        dict(
            max_num_steps = int(3e7),
            device = config["device"],
            entropy_coef=0.15,
            start_entropy = 0.15,
            final_entropy = 0.15,
            total_steps = 50000,
            num_steps_per_iter=512
        )
    )

    parameter_dicts = generate_parameter_combinations(
        config, 
        sweep_params=sweep_params,
        exp_name=experiment_name
    )

    submit_jobs(
        submit_fn=submit_fn,
        executor=executor,
        parameter_dicts=parameter_dicts,
        ask_permission=False,
        use_submitit=True
    )



if __name__ == "__main__":
    main()
