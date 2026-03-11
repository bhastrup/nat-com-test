import json

from torch.optim import Adam

from src.tools import util
from src.performance.energetics import str_to_EnergyUnit
from src.tools.arg_parser_pretrain import build_default_argparser_pretrain
from src.tools.arg_parser import build_default_argparser
from src.tools.model_util import get_model
from src.tools.env_util import EnvMaker
from src.performance.single_cpkt.evaluator import EvaluatorIO, SingleCheckpointEvaluator
from src.performance.cumulative.discovery_logger import CumulativeDiscoveryTracker
from src.rl.losses import EntropySchedule, RewardCoefficientSchedule
from scripts.train.pretrain_bc import pretrain_agent


def pretrain(config: dict) -> None:
    if 'reward_coefs' not in config:
        config['reward_coefs'] = {'rew_rae': 1.0 , 'rew_valid': 0.1}

    config_ft = config['config_ft']
    config['name'] = 'pretrain'
    tag = util.get_tag(config)
    util.set_seeds(seed=config['seed'])
    device = util.init_device(config['device'])
    util.create_directories([config['log_dir'], config['model_dir'], config['results_dir']])
    util.setup_logger(config, directory=config['log_dir'], tag=tag)

    if 'resave_config' not in config or config.get('resave_config'):
        util.save_config(config, directory=config['log_dir'], tag=tag)

    config['energy_unit'] = str_to_EnergyUnit(config['energy_unit'])

    # Load data and environments
    env_maker = EnvMaker(
        cf=config, 
        split_method=config['split_method']
    )
    training_envs, eval_envs, eval_envs_big = env_maker.make_envs()
    observation_space, action_space = env_maker.get_spaces()
    benchmark_energies = env_maker.ref_data.get_mean_energies()
    eval_formulas = util.get_str_formulas_from_vecenv(eval_envs)


    # Build model
    model, start_num_iter, model_handler, var_counts = get_model(
        config=config, observation_space=observation_space, action_space=action_space, device=device, tag=tag
    )

    # Build optimizer
    optimizer_online = Adam(model.parameters(), lr=config['learning_rate'],
                            amsgrad=True if config_ft['optimizer']=='amsgrad' else False)
    # data_loader = get_pretrain_dataloader(df_train, model, observation_space, action_space, config)

    # Discovery trackers
    logger = CumulativeDiscoveryTracker(
        cf=config,
        model=model,
        env_container_train=training_envs,
        env_container_eval=eval_envs,
        start_num_iter=start_num_iter,
    )

    # Build evaluator
    evaluator = SingleCheckpointEvaluator(
        eval_envs=eval_envs_big,
        reference_smiles=env_maker.get_reference_smiles(eval_formulas),
        benchmark_energies=benchmark_energies,
        io_handler=EvaluatorIO(base_dir=config['results_dir']),
        wandb_run = logger.wandb_run,
        num_episodes_const=None,
        prop_factor=1,
        calc_dipole=True if 'reward_coefs' in config and 'rew_dipole' in config['reward_coefs'] else False
    )

    es = config_ft["entropy_schedule"]
    entropy_schedule = EntropySchedule(
        start_value=es["start_value"],
        final_value=es["final_value"],
        start_iter=es["start_iter"],
        end_iter=es["end_iter"],
    )

    reward_coef_schedule = None
    if "reward_coef_schedule" in config_ft:
        rcs = config_ft["reward_coef_schedule"]
        reward_coef_schedule = RewardCoefficientSchedule(
            schedules=rcs["schedules"],
            start_iter=rcs["start_iter"],
            end_iter=rcs["end_iter"],
        )
    training_reward = env_maker.get_training_reward()

    # Train model
    total_num_iter = start_num_iter
    for epoch in range(config['num_epochs']):

        print(f"Entering pretrain_agent")
        total_num_steps = pretrain_agent(
            total_num_iter=total_num_iter,
            ac=model,
            optimizer_online=optimizer_online,
            mini_batch_size=config['mini_batch_size'],
            device=device,
            model_handler=model_handler,
            save_freq=config['save_freq'],
            eval_freq=config['eval_freq'],
            eval_envs=eval_envs,
            config=config,
            config_ft=config_ft,
            train_envs_online=training_envs,
            rl_algo_online=config['rl_algo_online'],
            logger=logger,
            info_saver=util.InfoSaver(directory=config['results_dir'], tag=tag),
            evaluator=evaluator,
            entropy_schedule=entropy_schedule,
            reward_coef_schedule=reward_coef_schedule,
            reward=training_reward,
        )

        print(f"Finished epoch {epoch} with {total_num_steps} steps")
        model_handler.save_after_full_replica(module=model, num_steps=total_num_steps, epochs=epoch)
        if total_num_steps > model_handler._checkpoints[-1]:
            break
        epoch += 1


    print(f"Finished training with {total_num_steps} steps. Saved checkpoints at {model_handler._checkpoints} steps.")
