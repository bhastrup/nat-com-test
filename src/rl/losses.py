
from typing import Dict, Tuple
import torch

from src.agents.base import AbstractActorCritic
from src.tools.util import to_numpy



def compute_loss_BC(
    ac: AbstractActorCritic,
    data: dict,
    device=None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    pred = ac.step(data['obs'], data['act'])
    bc_loss = - pred['logp'].mean()

    info = dict(
        bc_loss=to_numpy(bc_loss).item(),
        bc_logp=to_numpy(pred['logp']).mean(),
    )

    return bc_loss, info



def compute_loss_MARWIL(
    ac: AbstractActorCritic,
    data: dict,
    vf_coef: float=0.5,
    entropy_coef: float=0.01,
    beta: float=0.5,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    #print(f'data["obs"]: {data["obs"]}')
    #print(f'data["act"].shape: {data["act"].shape}, data["act"]: {data["act"]}')

    pred = ac.step(data['obs'], data['act'])

    # adv = torch.as_tensor(data['adv'], device=device)
    ret = torch.as_tensor(data['ret'], device=device)

    # Imitation loss (MARWIL)
    with torch.no_grad():
        adv_raw = ret - pred['v']
        adv = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)

    imit_loss = - (torch.exp(beta * adv) * pred['logp']).mean()

    # Entropy loss
    entropy_loss = -entropy_coef * pred['ent'].mean()

    # Value loss
    vf_loss = vf_coef * (pred['v'] - ret).pow(2).mean()

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
        M_mean_exp_bet_adv=to_numpy(torch.exp(beta * adv)).mean(),
        M_max_exp_bet_adv=to_numpy(torch.exp(beta * adv)).max(),
    )

    return total_loss, info


def compute_loss(
    ac: AbstractActorCritic,
    data: dict,
    clip_ratio: float,
    vf_coef: float,
    entropy_coef: float,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred = ac.step(data['obs'], data['act'])

    old_logp = torch.as_tensor(data['logp'], device=device)
    adv = torch.as_tensor(data['adv'], device=device)
    ret = torch.as_tensor(data['ret'], device=device)

    # Policy loss
    ratio = torch.exp(pred['logp'] - old_logp)
    obj = ratio * adv
    clipped_obj = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    policy_loss = -torch.min(obj, clipped_obj).mean()

    # Entropy loss
    entropy_loss = -entropy_coef * pred['ent'].mean()

    # Value loss
    vf_loss = vf_coef * (pred['v'] - ret).pow(2).mean()

    # Total loss
    loss = policy_loss + entropy_loss + vf_loss

    # Approximate KL for early stopping
    approx_kl = (old_logp - pred['logp']).mean()

    # Extra info
    clipped = ratio.lt(1 - clip_ratio) | ratio.gt(1 + clip_ratio)
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



import time, logging
from typing import Sequence

from torch.optim.optimizer import Optimizer

import numpy as np

from src.rl.buffer import DynamicPPOBuffer, get_batch_generator, collect_data_batch, compute_mean_dict
from src.tools.util import compute_gradient_norm




# Train policy with multiple steps of gradient descent
def train(
    ac: AbstractActorCritic,
    optimizer: Optimizer,
    data: Dict[str, Sequence],
    mini_batch_size: int,
    clip_ratio: float,
    target_kl: float,
    vf_coef: float,
    entropy_coef: float,
    gradient_clip: float,
    max_num_steps: int,
    device=None,
    rl_algo: str='PPO'
) -> dict:
    infos = {}

    start_time = time.time()

    num_epochs = 0

    max_num_steps = 1 if rl_algo == 'MARWIL' else max_num_steps
    for i in range(max_num_steps):
        optimizer.zero_grad()

        batch_infos = []
        batch_generator = get_batch_generator(indices=np.arange(len(data['obs'])), batch_size=mini_batch_size)
        for batch_indices in batch_generator:
            data_batch = collect_data_batch(data, indices=batch_indices)

            if rl_algo == 'PPO':
                batch_loss, batch_info = compute_loss(ac,
                                                      data=data_batch,
                                                      clip_ratio=clip_ratio,
                                                      vf_coef=vf_coef,
                                                      entropy_coef=entropy_coef,
                                                      device=device)
            elif rl_algo == 'MARWIL':
                batch_loss, batch_info = compute_loss_MARWIL(ac,
                                                             data=data_batch,
                                                             device=device,
                                                             vf_coef=vf_coef,
                                                             entropy_coef=entropy_coef)


            batch_loss.backward(retain_graph=False)  # type: ignore
            batch_infos.append(batch_info)

        loss_info = compute_mean_dict(batch_infos)
        loss_info['grad_norm'] = compute_gradient_norm(ac.parameters())

        # Check KL
        if rl_algo == 'PPO' and loss_info['approx_kl'] > 1.5 * target_kl:
            logging.debug(f'Early stopping at step {i} for reaching max KL.')
            break

        # Take gradient step
        logging.debug('Taking gradient step')
        torch.nn.utils.clip_grad_norm_(ac.parameters(), max_norm=gradient_clip)
        optimizer.step()
        optimizer.zero_grad()

        num_epochs += 1

        # Logging
        logging.debug(f'Online {rl_algo} loss {i}: {loss_info}')
        infos.update(loss_info)

    infos['num_opt_steps'] = num_epochs
    infos['time'] = time.time() - start_time

    # if num_epochs > 0:
    #     logging.info(f'Optimization: policy loss={infos["policy_loss"]:.3f}, vf loss={infos["vf_loss"]:.3f}, '
    #                  f'entropy loss={infos["entropy_loss"]:.3f}, total loss={infos["total_loss"]:.3f}, '
    #                  f'num steps={num_epochs}')
    return infos


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