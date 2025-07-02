from copy import deepcopy

import logging
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
import torch
from src.agents.base import AbstractActorCritic
from src.performance.cumulative.performance_summary import Logger, MultibagLogger, FinetuneLogger
from src.performance.single_cpkt.evaluator import SingleCheckpointEvaluator, launch_eval_jobs
from src.rl.buffer_container import PPOBufferContainer, PPOBufferContainerDeploy
from src.rl.env_container import VecEnv
from src.rl.rl_algos import PolicyOptimizer
from src.rl.rollouts import batch_rollout_with_logging, rollout_n_eps_per_env
from src.tools.model_util import ModelIO
from src.tools.util import RolloutSaver, InfoSaver



def lightweight_eval(eval_envs, ac, gamma, lam, logger, info_saver, rollout_saver, save_eval_rollout, total_num_iter):
    if eval_envs is None:
        return

    eval_container = PPOBufferContainerDeploy(size=eval_envs.get_size(), gamma=gamma, lam=lam)

    with torch.no_grad():
        # ac.training = False
        eval_rollout = rollout_n_eps_per_env(ac,
                                             eval_envs,
                                             buffer_container=eval_container,
                                             num_episodes=1,
                                             output_trajs=False,
                                             render=True)
        # ac.training = True
    eval_buffer = eval_container.merge()
    if logger:
        logger.save_rollout_and_info(info_saver=info_saver, rollout_saver=rollout_saver, rollout=eval_rollout, 
                                              save_rollout=save_eval_rollout, buffer=eval_buffer, name='eval', 
                                              total_num_iter=total_num_iter)
    
    print(f'Eval: {eval_rollout["return_mean"]:.2f} +- {eval_rollout["return_std"]:.2f}')




def merge_data(dict1: dict, dict2: dict):
    return {k: dict1[k] + dict2[k] if k == 'obs' 
            else np.concatenate([dict1[k], dict2[k]], axis=0) for k in dict2.keys()}


class Trainer:
    def __init__(
            self, 
            total_num_iter: int,
            ac: AbstractActorCritic,
            data_loader: DataLoader = None,
            policy_optimizer_offline: PolicyOptimizer = None,
            policy_optimizer_online: PolicyOptimizer = None,
            model_handler: Optional[ModelIO] = None,
            save_freq: int = 50,
            eval_freq: int = 2000,
            eval_envs: VecEnv = None,
            config: dict = None,
            config_ft: dict = None,
            train_envs_online: VecEnv = None,
            logger: Logger = None,
            evaluator: SingleCheckpointEvaluator = None,
            info_saver: Optional[InfoSaver] = None,
            rollout_saver: Optional[RolloutSaver] = None
    ):
        self.total_num_iter = total_num_iter
        self.ac = ac
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)
        self.policy_optimizer_offline = policy_optimizer_offline
        self.policy_optimizer_online = policy_optimizer_online
        self.model_handler = model_handler
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.eval_envs = eval_envs
        self.config = config
        self.config_ft = config_ft
        self.train_envs_online = train_envs_online
        self.logger = logger
        self.evaluator = evaluator
        self.info_saver = info_saver
        self.rollout_saver = rollout_saver

        self.burn_in = (15, 20)

        self.determine_training_flags()
        self.determine_target_iter()


    def determine_training_flags(self):
        """Determine if we are doing offline and/or online learning"""

        # Offline learning
        if self.policy_optimizer_offline is not None:
            assert self.data_loader is not None, "Offline learning requires a data loader"
            self.pretrain = True
        else:
            self.pretrain = False
            assert self.data_loader is None, "Online learning does not require a data loader"

        # Online learning
        if self.policy_optimizer_online is not None:
            assert self.train_envs_online is not None, "Online learning requires a training environment"
            self.online_learning = True
        else:
            self.online_learning = False

    def determine_target_iter(self):
        if self.online_learning:
            self.n_iter_online = self.config_ft["max_num_steps"] // self.config_ft["num_steps_per_iter"]
        else:
            self.n_iter_online = 0

        if self.pretrain:
            self.n_iter_pretrain = len(self.data_loader) * self.config['num_epochs']
        else:
            self.n_iter_pretrain = 0

        self.total_iter_target = max(self.n_iter_online, self.n_iter_pretrain)


    def make_pretraining_step_now(self) -> bool:
        return (self.total_num_iter % self.config['pretrain_every_k_iter'] == 0) \
            if self.config['pretrain_every_k_iter'] else False


    def collect_online_data(self) -> Tuple[dict, dict]:
        if self.online_learning == False or self.total_num_iter < self.burn_in[0]:
            return None, None
        
        online_container = PPOBufferContainer(size=self.train_envs_online.get_size(), gamma=1., lam=0.97)
        online_rollout = batch_rollout_with_logging(ac=self.ac, envs=self.train_envs_online,
                                                    buffer_container=online_container,
                                                    num_steps=self.config_ft['num_steps_per_iter'],
                                                    logger=self.logger)
        online_buffer = online_container.merge()
    
        if self.logger:
            self.logger.save_rollout_and_info(
                info_saver=self.info_saver,
                rollout_saver=self.rollout_saver,
                rollout=online_rollout,
                save_rollout=False,
                buffer=online_buffer,
                name='train', 
                total_num_iter=self.total_num_iter
            )
        
        online_data = online_buffer.get_data()

        return online_data, online_rollout
    

    def collect_offline_data(self):
        if not self.pretrain or not self.make_pretraining_step_now():
            return None

        try:
            return next(self.data_iter)
        except StopIteration:
            # Reset the iterator when it's exhausted
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)


    def optimization_step(self, online_data: dict, offline_data: dict, infos: dict):

        if online_data is not None:
            loss_info_online = self.policy_optimizer_online.optimize(
                data=online_data, mode='online', burn_in=self.total_num_iter < self.burn_in[1]
            )
            infos.update({'rl': loss_info_online})

        if offline_data is not None:
            loss_info_offline = self.policy_optimizer_offline.optimize(offline_data, mode='offline')
            infos.update({'pretrain': loss_info_offline})

        return infos


    def checkpoint_saving(self):
        if not self.model_handler:
            return
        
        # Save model (perpetually)
        if self.total_num_iter % self.save_freq == 0:
            self.model_handler.save(self.ac, num_steps=self.total_num_iter)
    
        # Save model (big checkpoint)
        if self.total_num_iter in self.model_handler._checkpoints:
            cp_path = self.model_handler.save_if_checkpoint(self.ac, self.total_num_iter)


    def log_infos(self, infos: dict):
        if not self.logger:
            return
        
        infos['total_num_iter'] = self.total_num_iter
        print(infos)

        if self.logger.wandb_run is not None:
            self.logger.wandb_run.log(infos)
    

    def evaluation(self):
        if self.eval_envs is None:
            return

        # Single shot evaluation: No longer used in finetuning_logger besides simple logging
        if self.total_num_iter % self.config['eval_freq_fast'] == 0:
            print(f"Fast evaluation at {self.total_num_iter} steps")
            lightweight_eval(
                eval_envs=deepcopy(self.eval_envs), 
                ac=self.ac, 
                gamma=1., 
                lam=0.97, 
                logger=self.logger,
                info_saver=None, 
                rollout_saver=None, 
                save_eval_rollout=False,
                total_num_iter=self.total_num_iter
            )

        # Evaluate
        if (self.total_num_iter > 0) and (self.total_num_iter % self.eval_freq == 0):
            launch_eval_jobs(
                deepcopy(self.ac),
                evaluator=self.evaluator,
                n_batches=self.total_num_iter,
                batch_size=self.config_ft["num_steps_per_iter"],
                cf=self.config,
            )
    
    def increment_total_num_iter(self):
        self.total_num_iter += 1
        if self.logger:
            self.logger.increment_total_num_iter()


    def train(self):

        # Training loop
        start_iter = self.total_num_iter
        assert start_iter < self.total_iter_target, "Start iter is greater than the number of loops"

        for i in range(start_iter, self.total_iter_target):
            infos = {}

            # Collect data
            online_data, online_rollout = self.collect_online_data()
            offline_data = self.collect_offline_data()
            assert offline_data is not None or online_data is not None, \
                "We have to do either online or offline learning"

            # Optimize
            infos = self.optimization_step(online_data, offline_data, infos)

            # Log infos
            self.log_infos(infos)

            # Play evaluation rollouts
            self.evaluation()

            # Save checkpoint
            self.checkpoint_saving()

            # Increment total num iter
            self.increment_total_num_iter()

            # Break if we have reached the last checkpoint
            if self.total_num_iter >= self.model_handler._checkpoints[-1]:
                break

        logging.info(f"Finished pretraining with {self.total_num_iter} steps.")
        logging.info(f"Saved checkpoints at {self.model_handler._checkpoints} steps.")
