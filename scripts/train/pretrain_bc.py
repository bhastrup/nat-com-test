import logging, time
from typing import Optional, Any
from copy import deepcopy

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.tools.model_util import ModelIO
from src.tools.util import RolloutSaver, InfoSaver, compute_gradient_norm
from src.agents.base import AbstractActorCritic
from src.rl.env_container import VecEnv
from src.rl.buffer_container import PPOBufferContainer, PPOBufferContainerDeploy
from src.rl.buffer import get_batch_generator
from src.rl.rollouts import batch_rollout_with_logging, rollout_n_eps_per_env
from src.rl.losses import (
    compute_loss, compute_loss_MARWIL, compute_loss_BC, train, EntropySchedule
)
from src.performance.single_cpkt.evaluator import SingleCheckpointEvaluator, launch_eval_jobs
from src.performance.cumulative.performance_summary import Logger, MultibagLogger, FinetuneLogger


def optimize_agent(ac: AbstractActorCritic, data_batch: dict, optimizer: Optimizer, 
                   vf_coef: float, entropy_coef: float,  beta: float, device: torch.device, 
                   gradient_clip: float = 0.5, rl_algo_pretrain: str = 'PPO', grad_steps_offline: int=1):

    if rl_algo_pretrain == 'PPO':
        loss_info = optimize_agent_PPO(ac=ac, data_batch=data_batch, optimizer=optimizer,
                            vf_coef=vf_coef, entropy_coef=entropy_coef, beta=beta, device=device,
                            gradient_clip=gradient_clip)
    elif rl_algo_pretrain == 'MARWIL':
        loss_info = optimize_agent_MARWIL(ac=ac, data_batch=data_batch, optimizer=optimizer,
                            vf_coef=vf_coef, entropy_coef=entropy_coef, beta=beta, device=device,
                            gradient_clip=gradient_clip, grad_steps_offline=grad_steps_offline)
    elif rl_algo_pretrain == 'bc':
        loss_info = optimize_agent_bc(ac=ac, data_batch=data_batch, optimizer=optimizer,
                            device=device, gradient_clip=gradient_clip, grad_steps=grad_steps_offline)
    else:
        raise ValueError(f'Unknown RL algorithm: {rl_algo_pretrain}')
    
    return loss_info
    


def optimize_agent_MARWIL(ac: AbstractActorCritic, data_batch: dict, optimizer: Optimizer, 
                          vf_coef: float, entropy_coef: float,  beta: float, device: torch.device, 
                          gradient_clip: float = 0.5, grad_steps_offline: int = 2):
    
    ac.training = True

    for _ in range(grad_steps_offline):
        optimizer.zero_grad()
        loss, loss_info = compute_loss_MARWIL(ac, 
                                              data=data_batch,
                                              vf_coef=vf_coef,
                                              entropy_coef=entropy_coef,
                                              beta=beta,
                                              device=device)
        
        loss.backward()
        loss_info['grad_norm'] = compute_gradient_norm(ac.parameters())
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()

    return loss_info


def optimize_agent_bc(ac: AbstractActorCritic, data_batch: dict, optimizer: Optimizer, 
                      device: torch.device, gradient_clip: float = 0.5, grad_steps: int = 2):

    ac.training = True

    for _ in range(grad_steps):
        optimizer.zero_grad()
        loss, loss_info = compute_loss_BC(ac,
                                          data=data_batch,
                                          device=device)
        
        loss.backward()
        loss_info['grad_norm'] = compute_gradient_norm(ac.parameters())
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()

    return loss_info


def optimize_agent_PPO(ac: AbstractActorCritic, data_batch: dict, optimizer: Optimizer, 
                   vf_coef: float, entropy_coef: float,  beta: float, device: torch.device, 
                   gradient_clip: float = 0.5):
    
    target_kl = 0.05
    clip_ratio = 0.2
    max_num_steps = 7

    ac.training = True
    infos = {}
    start_time = time.time()


    with torch.no_grad():
        data_batch['logp'] = ac.step(data_batch['obs'], data_batch['act'])['logp']


    num_epochs = 0
    for i in range(max_num_steps):
        optimizer.zero_grad()
        loss, loss_info = compute_loss(ac, 
                                      data=data_batch,
                                      clip_ratio=clip_ratio,
                                      vf_coef=vf_coef,
                                      entropy_coef=entropy_coef,
                                      device=device)
        loss.backward(retain_graph=False)

        loss_info['grad_norm'] = compute_gradient_norm(ac.parameters())

        # Check KL
        if loss_info['approx_kl'] > 1.5 * target_kl:
            logging.debug(f'Early stopping at step {i} for reaching max KL.')
            break

        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()

        num_epochs += 1

        # Logging
        logging.debug(f'Loss {i}: {loss_info}')
        infos.update(loss_info)

    infos['num_opt_steps'] = num_epochs
    infos['time'] = time.time() - start_time


    return loss_info


def merge_data(dict1: dict, dict2: dict):
    return {k: dict1[k] + dict2[k] if k == 'obs' 
            else np.concatenate([dict1[k], dict2[k]], axis=0) for k in dict2.keys()}


def lightweight_eval(eval_envs, ac, gamma, lam, logger, info_saver, rollout_saver, save_eval_rollout, total_num_iter):

    if eval_envs is None:
        return

    eval_container = PPOBufferContainerDeploy(size=eval_envs.get_size(), gamma=gamma, lam=lam)

    with torch.no_grad():
        ac.training = False # argmax
        eval_rollout = rollout_n_eps_per_env(ac,
                                             eval_envs,
                                             buffer_container=eval_container,
                                             num_episodes=1,
                                             output_trajs=False)
        ac.training = True
    eval_buffer = eval_container.merge()
    if logger:
        # Watch not to use same logger as main logger
        logger.save_rollout_and_info(info_saver=info_saver, rollout_saver=rollout_saver, rollout=eval_rollout, 
                                     save_rollout=save_eval_rollout, buffer=eval_buffer, name='eval', 
                                     total_num_iter=total_num_iter)
    
    print(f'Eval: {eval_rollout["return_mean"]:.2f} +- {eval_rollout["return_std"]:.2f}')



def pretrain_agent(
    total_num_iter: int,
    ac: AbstractActorCritic,
    optimizer_online: Optimizer,
    mini_batch_size: int,
    device: torch.device,
    save_freq: int = 50,
    model_handler: Optional[ModelIO] = None,
    eval_freq: int = 2000,
    eval_envs: VecEnv = None,
    config: dict = None,
    config_ft: dict = None,
    train_envs_online: VecEnv = None,
    rl_algo_online: str = 'PPO',
    logger: Logger = None,
    rollout_saver: Optional[RolloutSaver] = None,
    info_saver: Optional[InfoSaver] = None,
    evaluator: SingleCheckpointEvaluator = None,
    entropy_schedule: EntropySchedule = None
):
    assert rl_algo_online == 'PPO', "Unknown online RL algorithm"


    num_training_iterations = config_ft["max_num_steps"] // config_ft["num_steps_per_iter"]
    for i in range(num_training_iterations):
        infos = {}

        # Save model (big checkpoint)
        if model_handler:
            if total_num_iter in model_handler._checkpoints:
                cp_path = model_handler.save_if_checkpoint(ac, total_num_iter)

        ####  Collect data online
        if rl_algo_online:
            online_container = PPOBufferContainer(size=train_envs_online.get_size(), gamma=1., lam=0.97)
            online_rollout = batch_rollout_with_logging(ac=ac, envs=train_envs_online,
                                                        buffer_container=online_container,
                                                        num_steps=config_ft['num_steps_per_iter'],
                                                        logger=logger)
            online_buffer = online_container.merge()
            if logger:
                logger.save_rollout_and_info(
                    info_saver=info_saver,
                    rollout_saver=rollout_saver,
                    rollout=online_rollout, 
                    save_rollout=False,
                    buffer=online_buffer,
                    name='train', 
                    total_num_iter=total_num_iter
                )
            online_data = online_buffer.get_data()


        # Optimize
        if online_rollout['return_std'] != 0.0:
            print(f"starting optimization at step {total_num_iter}")
            opt_info_online = train(
                ac=ac,
                optimizer=optimizer_online,
                data=online_data,
                mini_batch_size=mini_batch_size,
                clip_ratio=config_ft['clip_ratio'],
                vf_coef=config_ft['vf_coef'],
                entropy_coef=entropy_schedule.calculate(step=total_num_iter) \
                    if entropy_schedule else config_ft['entropy_coef'],
                target_kl=config_ft['target_kl'],
                gradient_clip=config_ft['gradient_clip'],
                max_num_steps=config_ft['max_num_train_iters'],
                device=device,
                rl_algo='PPO',
            )
            infos.update(opt_info_online)


        if logger:
            # TODO: Move inside logger. What happens here actually?
            if logger.wandb_run is not None:
                infos['total_num_iter'] = total_num_iter
                logger.wandb_run.log(infos)
            

        # Save model (perpetually)
        if model_handler and (total_num_iter % save_freq == 0):
            model_handler.save(ac, num_steps=total_num_iter)
        
    
        # Stochastic evaluation
        if (total_num_iter > 0) and (total_num_iter % eval_freq == 0):
            if eval_envs is not None:
                launch_eval_jobs(
                    deepcopy(ac),
                    evaluator=evaluator,
                    n_batches=total_num_iter,
                    batch_size=config_ft["num_steps_per_iter"],
                    cf=config,
                )
        

        total_num_iter += 1
        if logger:
            logger.increment_total_num_iter()
        if total_num_iter >= model_handler._checkpoints[-1]:
            break


    return total_num_iter
