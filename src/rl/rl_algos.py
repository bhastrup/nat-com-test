import time, logging
from typing import Dict, Sequence, Tuple, Callable
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from src.agents.base import AbstractActorCritic
from src.rl.buffer import (
    DynamicPPOBuffer, 
    get_batch_generator, 
    collect_data_batch, 
    compute_mean_dict
)
from src.tools.util import compute_gradient_norm, to_numpy



class PolicyOptimizer(ABC):
    """Base class for all policy optimization algorithms."""

    tag: str = None
    type: str = None

    def __init__(
        self, 
        ac: AbstractActorCritic,
        optimizer: Optimizer,
        device: torch.device,
        gradient_clip: float = 0.5,
    ):
        self.ac = ac
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip

    @abstractmethod
    def compute_loss(self, data: dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss for the policy optimization algorithm."""
        raise NotImplementedError

    def optimize(self, data: dict, mode: str = 'offline') -> Dict[str, float]:
        """Optimize the agent with the given data batch."""
        self.ac.training = True
        self.optimizer.zero_grad()

        loss, loss_info = self.compute_loss(data)
        loss.backward()
        loss_info['grad_norm'] = compute_gradient_norm(self.ac.parameters())
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=self.gradient_clip)
        self.optimizer.step()
            
        return loss_info


def create_policy_optimizer(
    algo_online: str,
    algo_pretrain: str,
    ac: AbstractActorCritic,
    nn_optimizer_offline: Optimizer,
    nn_optimizer_online: Optimizer,
    device=None,
    config: dict = None,
    config_ft: dict = None
) -> Tuple[PolicyOptimizer, PolicyOptimizer]:
    """Factory function to create the appropriate policy optimizers based on algorithm name."""

    online_optimizers = [None, "ppo"]
    offline_optimizers = [None, "bc", "marwil"]

    if algo_online.lower() not in online_optimizers:
        raise ValueError(f"Unknown online algorithm: {algo_online}")
    if algo_pretrain.lower() not in offline_optimizers:
        raise ValueError(f"Unknown offline algorithm: {algo_pretrain}")


    # Offline optimizers
    if algo_pretrain is None:
        offline_optimizer = None
    elif algo_pretrain.lower() == 'bc':
        offline_optimizer = BC(
            ac=ac,
            optimizer=nn_optimizer_offline,
            device=device,
            gradient_clip=config['gradient_clip'],
        )
    elif algo_pretrain.lower() == 'marwil':
        offline_optimizer = MARWIL(
            ac=ac,
            optimizer=nn_optimizer_offline,
            vf_coef=config['vf_coef'],
            entropy_coef=config['entropy_coef'],
            beta=config['beta_MARWIL'],
            device=device,
            gradient_clip=config['gradient_clip'],
        )
    else:
        offline_optimizer = None

    # Online optimizers
    if algo_online is None:
        online_optimizer = None
    elif algo_online.lower() == 'ppo':
        online_optimizer = PPO(
            ac=ac,
            optimizer=nn_optimizer_online,
            clip_ratio=config_ft['clip_ratio'],
            vf_coef=config_ft['vf_coef'],
            entropy_coef=config_ft['entropy_coef'],
            target_kl=config_ft['target_kl'],
            device=device,
            gradient_clip=config_ft['gradient_clip'],
            max_num_steps=config_ft['max_num_train_iters'],
            mini_batch_size=config_ft['mini_batch_size'],
        )
    else:
        online_optimizer = None

    return offline_optimizer, online_optimizer




class BC(PolicyOptimizer):
    """Behavior Cloning policy optimizer."""    
    tag = 'BC'
    type = 'offline'

    def compute_loss(self, data: dict, mode: str = 'offline') -> Tuple[torch.Tensor, dict]:
        pred = self.ac.step(data['obs'], data['act'])
        bc_loss = -pred['logp'].mean()
        
        info = dict(
            bc_loss=to_numpy(bc_loss).item(),
            bc_logp=to_numpy(pred['logp']).mean(),
        )

        return bc_loss, info


class MARWIL(PolicyOptimizer):
    """MARWIL (Monotonic Advantage Re-Weighted Imitation Learning) policy optimizer."""
    tag = 'MARWIL'
    type = 'offline'

    def __init__(
        self,
        ac: AbstractActorCritic,
        optimizer: Optimizer,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        beta: float = 0.5,
        device: torch.device = None,
        gradient_clip: float = 0.5,
    ):
        super().__init__(ac, optimizer, device, gradient_clip)
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.beta = beta
    
    def compute_loss(self, data: dict, mode: str = 'offline') -> Tuple[torch.Tensor, dict]:
        pred = self.ac.step(data['obs'], data['act'])
        ret = torch.as_tensor(data['ret'], device=self.device)
        
        # Advantage computation
        with torch.no_grad():
            adv_raw = ret - pred['v']
            adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
        
        # Imitation loss (MARWIL)
        imit_loss = -(torch.exp(self.beta * adv) * pred['logp']).mean()
        
        # Entropy loss
        entropy_loss = -self.entropy_coef * pred['ent'].mean()
        
        # Value loss
        vf_loss = self.vf_coef * (pred['v'] - ret).pow(2).mean()
        
        # Total loss
        total_loss = imit_loss + entropy_loss + vf_loss
        
        info = dict(
            M_entropy_loss=to_numpy(entropy_loss).item(),
            M_vf_loss=to_numpy(vf_loss).item(),
            M_total_loss=to_numpy(total_loss).item(),
            M_imit_loss=to_numpy(imit_loss).item(),
            M_logp=to_numpy(pred['logp']).mean(),
            M_adv_raw_mean=to_numpy(adv_raw).mean(),
            M_adv_max=to_numpy(adv).max(),
            M_adv_min=to_numpy(adv).min(),
            M_adv_std=to_numpy(adv).std(),
            M_adv_norm_max=to_numpy(adv).max(),
            M_mean_exp_bet_adv=to_numpy(torch.exp(self.beta * adv)).mean(),
            M_max_exp_bet_adv=to_numpy(torch.exp(self.beta * adv)).max(),
        )
        
        return total_loss, info




class PPO(PolicyOptimizer):
    tag = 'PPO'
    type = 'online'
    """Proximal Policy Optimization policy optimizer."""
    def __init__(
        self,
        ac: AbstractActorCritic,
        optimizer: Optimizer,
        clip_ratio: float = 0.2, 
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        target_kl: float = 0.05,
        device=None,
        gradient_clip: float = 0.5,
        max_num_steps: int = 7,
        mini_batch_size: int = 128,
    ):
        super().__init__(ac, optimizer, device, gradient_clip)
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.max_num_steps = max_num_steps
        self.mini_batch_size = mini_batch_size

    def compute_loss(self, data: dict, burn_in: bool = False) -> Tuple[torch.Tensor, dict]:
        pred = self.ac.step(data['obs'], data['act'])
        
        old_logp = torch.as_tensor(data['logp'], device=self.device)
        adv = torch.as_tensor(data['adv'], device=self.device)
        ret = torch.as_tensor(data['ret'], device=self.device)
        
        # Policy loss
        ratio = torch.exp(pred['logp'] - old_logp)
        obj = ratio * adv
        clipped_obj = ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        policy_loss = -torch.min(obj, clipped_obj).mean()
        
        # Entropy loss
        entropy_loss = -self.entropy_coef * pred['ent'].mean()
        
        # Value loss
        vf_loss = self.vf_coef * (pred['v'] - ret).pow(2).mean()
        
        # Total loss
        if burn_in == True:
            loss = vf_loss
        else:
            loss = policy_loss + entropy_loss + vf_loss
        
        # Approximate KL for early stopping
        approx_kl = (old_logp - pred['logp']).mean()
        
        # Extra info
        clipped = ratio.lt(1 - self.clip_ratio) | ratio.gt(1 + self.clip_ratio)
        clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()
        
        info = dict(
            policy_loss=to_numpy(policy_loss).item(),
            entropy_loss=to_numpy(entropy_loss).item(),
            vf_loss=to_numpy(vf_loss).item(),
            total_loss_ppo=to_numpy(loss).item(),
            approx_kl=to_numpy(approx_kl).item(),
            clip_fraction=to_numpy(clip_fraction).item(),
        )
        
        return loss, info

    def optimize(self, data: dict, mode: str = 'online', burn_in: bool = False) -> Dict[str, float]:
        """PPO-specific optimization with KL divergence early stopping."""
        self.ac.training = True
        self.set_burn_in(burn_in=burn_in)

        infos = {}
        start_time = time.time()
                

        if mode == 'offline':
            # Crude attempt to use PPO in an offline manner using expert data
            with torch.no_grad():
                data['logp'] = self.ac.step(data['obs'], data['act'])['logp']
 

        num_opt_steps = 0
        for i in range(self.max_num_steps):
            self.optimizer.zero_grad()

            batch_infos = []
            batch_generator = get_batch_generator(indices=np.arange(len(data['obs'])), 
                                                  batch_size=self.mini_batch_size)
            for batch_indices in batch_generator:
                data_batch = collect_data_batch(data, indices=batch_indices)
                batch_loss, batch_info = self.compute_loss(data_batch, burn_in=burn_in)
                batch_loss.backward(retain_graph=False)
                batch_infos.append(batch_info)
            
            loss_info = compute_mean_dict(batch_infos)
            loss_info['grad_norm'] = compute_gradient_norm(self.ac.parameters())

            # Check KL
            if not burn_in and loss_info['approx_kl'] > 1.5 * self.target_kl:
                logging.debug(f'Early stopping at step {i} for reaching max KL.')
                break
            
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()
            
            num_opt_steps += 1
            
            # Logging
            logging.debug(f'Loss {i}: {loss_info}')
            infos.update(loss_info)
        
        infos['num_opt_steps'] = num_opt_steps
        infos['time'] = time.time() - start_time
        
        self.set_burn_in(burn_in=False)
        return infos
    
    def set_burn_in(self, burn_in: bool = True):
        if burn_in:
            # Freeze all parameters
            for param in self.ac.parameters():
                param.requires_grad = False

            # Unfreeze Critic parameters
            for param in self.ac.critic.parameters():
                param.requires_grad = True

        else:
            # Unfreeze all parameters
            for param in self.ac.parameters():
                param.requires_grad = True




class EntropySchedule:
    def __init__(self, start_entropy, final_entropy, total_steps):
        """
        Initializes the EntropySchedule.
        
        :param start_entropy: The initial entropy value at the start of training.
        :param final_entropy: The final entropy value at the end of training.
        :param total_steps: The total number of steps over which to transition from start_entropy to final_entropy.
        """
        self.start_entropy = start_entropy
        self.final_entropy = final_entropy
        self.total_steps = total_steps

    def calculate(self, step):
        """
        Calculate the entropy value at a given step.
        
        :param step: The step for which to calculate the entropy value.
        :return: The entropy value at the given step.
        """
        if step < self.total_steps:
            new_entropy = step / self.total_steps * (self.final_entropy - self.start_entropy) + self.start_entropy
            return new_entropy
        else:
            return self.final_entropy
