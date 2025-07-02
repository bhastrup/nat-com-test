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


"""
######################### GUIDE ###################################

REPRODUCE RESULTS FROM PAPER:
The hyperparameters shown here reproduce the results from the paper for the agent "A".
To setup the other agent trainings, you can simply copy-paste this file and change the 'wandb_group' to e.g. 'Agent-AV'
and specify the relevant 'reward_coefs' (see the other reward_coefs that are commented out below).
	

SUBMITIT:
This file is a submitit-based submission file that can be used to launch a single training job, or an array of training jobs.
In the sweep_params below, you can specify the hyperparameters you wish to sweep over by providing a list of parameter values.
However, beware that it creates a Cartesian product of all you sweep parameters (like a grid search) and submits an entire 
array of parameter dictionaries that will be submitted as SLURM jobs individually.

	
RERUNS:
To rerun a crashed job, specify the old checkpoint as shown in "A: Reruns".

COMPUTE:
Compute 


ARGUMENT STRUCTURE:
During our research, we experimented with pretraining-finetuning workflows, before focusing on the tabula rasa (from scratch)
setting for the paper. 

"""


def main() -> None:
    

    wandb_group = 'Agent-A' 			# Specify the name of this training, e.g. indicate which agent is used: A, AV, AFV, FV, F etc.
        
    
    # Section A: Reruns
    # In case you need to extend a training (e.g. due to a crash), provide the exact model checkpoint here and tell it to use the old config file.
    load_model = None 				# 'RUNS_FOLDER/EXPERIMENT_NAME/seed_1/models/CHECKPOINT_NAME.model'
    load_latest = False				# False
    use_old_config = False			# True

    config = load_default_configuration(use_old_config, load_model)
    if use_old_config and (load_model or load_latest):
        config['load_model'] = load_model
        config['load_latest'] = load_latest
        config['safe_xtb'] = False		# Can be set to True, if XTB for some reason caused everything to crash.
        submit_jobs(
            submit_fn=submit_fn,
            executor=None,
            parameter_dicts=[config],
            ask_permission=False,
            use_submitit=False
        )


    # Section B: Compute resources
    experiment_name = wandb_group
    dt_string = datetime.now().strftime("%d_%m_%H_%M")
    executor = submitit.AutoExecutor(folder=get_sublog_name(experiment_name, dt_string))
    if executor._executor.__class__.__name__ == 'SlurmExecutor':
        executor.update_parameters(
            slurm_partition="my_gpu_partition_name",		# Select the GPU partition on your university SLURM cluster.
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=8,
            slurm_nodes=1,
            slurm_time="7-00:00:00", # 7 days used here
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
    	# For basic testing/developing on your 'LocalExecuter' (i.e. your laptop):
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
    
    # Section C: Sweep parameter names and values
    sweep_params = dict(
        seed=[0, 1, 2]
    )


    # Section D: Other parameters to be included in all runs
    config.update(
        dict(
            wandb_project='isomer-discovery',
            train_mode='tabula_rasa',
            wandb_job_type='',
            entity='my_wandb_username', 	# Specify your username if you have an acocunt. Otherwise set save_to_wandb=False above.
            name=experiment_name,
            wandb_group=wandb_group,

            # Data
            mol_dataset='QM7',
            energy_unit='eV',

            # Environment
            min_reward=-3,
            safe_xtb=False,

	    # Specify reward coefficients
            reward_coefs = {'rew_atomisation': 1.0} 							# A
            #reward_coefs = {'rew_formation': 1.0} 							# F
            #reward_coefs = {'rew_atomisation': 1.0, 'rew_valid': 3.0} 					# AV
            #reward_coefs = {'rew_formation': 1.0, 'rew_valid': 3.0}					# FV
            #reward_coefs = {'rew_atomisation': 1.0, 'rew_formation': 1.0, 'rew_valid': 3.0} 		# AFV


            # Agent policy
            model = 'painn',
            rms_norm_update = False,

            # Model IO
            device='cuda',
            checkpoints = [int(c) for c in np.arange(0, 51) * int(2500)],
            load_model = load_model,

            # Training scheme
            rl_algo_online='PPO',
        )
    )

    # Section E: Online RL parameters
    config["config_ft"].update(
        dict(
            max_num_steps = int(3e7),
            device = config["device"],
            entropy_coef=0.15,
            start_entropy = 0.15,
            final_entropy = 0.25,
            total_steps = 30000,
            num_steps_per_iter=512
        )
    )


    # Section F: Create 
    parameter_dicts = generate_parameter_combinations(
        config, 
        sweep_params=sweep_params,
        exp_name=experiment_name
    )

    submit_jobs(
        submit_fn=submit_fn,
        executor=executor,
        parameter_dicts=parameter_dicts,
        ask_permission=False, # If True, it will ask you to verify the submission before it either submits it via SLURM or executes it directly.
        use_submitit=True
    )



if __name__ == "__main__":
    main()

