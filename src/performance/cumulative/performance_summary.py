import dataclasses, pickle, os, logging
from datetime import datetime
from typing import List, Dict, Tuple, Union, Any

from ase import Atoms
import gym
import numpy as np
import wandb

from src.rl.spaces import ObservationType, ObservationSpace
from src.tools import util
# from molgym.pretraining.dataset_torch import process_molecule
from src.rl.buffer import DynamicPPOBuffer, compute_buffer_stats

from src.performance.energetics import EnergyUnit, XTBOptimizer
from src.agents.base import AbstractActorCritic
from src.agents.painn.agent_painn_internal_multimodal import PainnInternalMultiModal
from src.agents.painn.new_model import NewAC
from src.agents.painn.agent import PainnAC
from src.tools.util import count_vars


class Logger():
    """Base class for logging training and evaluation metrics"""
    def __init__(
        self, 
        cf: dict = None, 
        model: AbstractActorCritic = None,
        start_num_iter: int = 0
    ) -> None:

        self.cf = cf
        self._init_wandb(cf, model)

        # Number of steps per iteration
        if 'num_steps_per_iter' in cf:
            self.num_steps_per_iter = cf['num_steps_per_iter']
        elif 'config_ft' in cf and 'num_steps_per_iter' in cf['config_ft']:
            self.num_steps_per_iter = cf['config_ft']['num_steps_per_iter']
        else:
            raise ValueError('num_steps_per_iter not found in config')

        # Initialize counters
        self.n_eps_current_rollout = 0
        self.total_num_iter = start_num_iter
        # self.num_env_steps = self.num_steps_per_iter * start_num_iter
    
    def increment_total_num_iter(self):
        self.total_num_iter += 1

    def get_num_steps(self):
        return self.num_steps_per_iter * self.total_num_iter
    
    # def _increase_step_count(self, num_steps: int) -> None:
    #     self.num_env_steps += num_steps

    def _init_wandb(self, cf: dict, model: AbstractActorCritic) -> None:
        var_counts = count_vars(model)
        logging.info(f'Number of parameters: {var_counts}')

        if cf['save_to_wandb']:

            now = datetime.now()
            dt_string = now.strftime("%d_%m_%H_%M")
            cf['wandb_name'] = cf['wandb_name'] + '_' + dt_string

            wandb_run = wandb.init(
                config=cf,
                mode=cf['wandb_mode'],
                entity=cf["entity"],
                project=cf['wandb_project'],
                group=cf['wandb_group'],
                name=cf['wandb_name'],
                job_type=cf['wandb_job_type'],
                reinit=True
            )

            if cf['wandb_watch_model'] == True and model is not None:
                wandb_run.watch(model, log="gradients", log_freq=1000)

        self.wandb_run = wandb_run if cf['save_to_wandb'] else None

    def _simple_rollout_logging(self, rollout, name: str = 'train'):
        if name == 'train':
            rollout_name = 'Training rollout'
        elif name == 'eval':
            rollout_name = 'Evaluation rollout'

        logging.info(
            f'{rollout_name}: return={rollout["return_mean"]:.3f} ({rollout["return_std"]:.1f}), '
            f'episode length={rollout["episode_length_mean"]:.1f}')
        
        if self.wandb_run:
            rollout = {f'{k}_{name}': v for k, v in rollout.items()}
            rollout.update({
                'total_num_iter': self.total_num_iter, 
                'num_env_steps': self.get_num_steps()}
            )
            self.wandb_run.log(rollout)


    def save_rollout_and_info(self, info_saver: util.InfoSaver, rollout_saver: util.RolloutSaver, 
                              save_rollout: bool, rollout: dict, buffer: DynamicPPOBuffer, 
                              name: str, total_num_iter: int):
        
        total_num_steps = total_num_iter * self.num_steps_per_iter

        self._simple_rollout_logging(rollout, name=name)

        if info_saver:
            rollout['total_num_steps'] = total_num_steps
            rollout.update(compute_buffer_stats(buffer))
            info_saver.save(rollout, name=name)


        if rollout_saver and save_rollout:
            rollout_saver.save(buffer, num_steps=total_num_steps, info=name)


    def save_optimization_data(self, info_saver: util.InfoSaver, opt_info: dict, total_num_steps: int):
        if info_saver:
            opt_info['total_num_steps'] = total_num_steps
            info_saver.save(opt_info, name='opt')
            if self.wandb_run is not None:
                try:
                    self.wandb_run.log(opt_info)
                except Exception as e:
                    print(f"An error occurred with wandb log opt_info: {e}")


    def save_episode_RL(self, state: ObservationType, total_reward: float, info: dict, name: str) -> None:
        pass


    def update_and_log_data(self, total_num_iter: int) -> None:
        pass



@dataclasses.dataclass
class MolCandidate:
    atoms: ObservationType
    energy: float
    reward: float
    relaxed_energy: float
    num_episodes_rollout: int


class FinetuneLogger(Logger):
    """Logger for singlebag finetuning"""
    def __init__(self, env: gym.Env, output_dir: str, wandb_run: Any = None, relax_best: bool = True):
        assert len(env.formulas) == 1, 'FinetuneLogger only works for single-task environments'
        super().__init__()

        self.energy_unit = EnergyUnit.EV

        raise NotImplementedError('FinetuneLogger is not compatibale with new logging setup')

        self.env = env
        self.output_dir = output_dir
        self.wandb_run = wandb_run
        self.relax_best = relax_best

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Benchmark
        self.n_atoms_target = util.get_formula_size(env.formulas[0])
        self.benchmark_energy = self.env.benchmark_energy
        
        # self.benchmark_return = 
        

        # Running metrics
        self.total_episodes_train = 0
        self.total_episodes_train_vec = [0]

        self.RL_total_rewards = []
        self.RL_total_rewards_avg = []
        self.RL_total_rewards_std = []

        self.RL_energy = []
        self.RL_energy_avg = []
        self.RL_energy_std = []

        num_termination_types = 3 # len(self.env.num_termination_types) # full_molecule, invalid_action, stop_token
        self.RL_info = []
        self.RL_info_count = [[] for _ in range(num_termination_types)]

        self.eval_total_rewards = []
        self.eval_total_energy = []
        self.best_eval_energy = np.inf
        self.best_eval_reward = -np.inf

        self.candidate_list: List[MolCandidate] = []
        self.best_eval_energy_relax = np.inf
        
        # Best metrics
        # self.best_eval_reward = -np.inf
        self.RL_best_reward = 0
        self.RL_best_energy = self.benchmark_energy+5
        self.best_num_atoms = 0
        self.RL_best_energy_vec = []

        self.num_episodes_rollout = 0
        self.RL_episodes_count = 0
        self.best_count = -1
        self.mols_built = 0


    def save_episode_RL(self, state: ObservationType, total_reward: float, info: dict, name: str) -> None:
        # TODO: keep the best molecules in a list and save them to a file/ase traj
        
        self.num_episodes_rollout += 1

        # self.RL_episodes_count += 1
        if info['termination_info'] == 'full_formula':
            self.mols_built += 1

        self.RL_info.append(info['termination_info'])

        num_atoms = len(self.observation_space.parse(state)[0])
        energy = info['energy'] if num_atoms == self.n_atoms_target else np.nan

        print(f"Episode {self.num_episodes_rollout} | {info['termination_info']} | total_reward={total_reward:.2f} | num_atoms={num_atoms} | name={name}")

        if name == 'eval':
            print(f"total_reward={total_reward:.2f}, self.best_eval_reward={self.best_eval_reward:.2f}")
            self.eval_total_rewards.append(total_reward)
            if energy is not np.nan:
                self.eval_total_energy.append(energy)
            if total_reward > self.best_eval_reward:
                self.best_count += 1
                self.best_eval_reward = total_reward
                self.best_eval_energy = energy

                self.best_state_eval = state
                self.best_molecule_eval = self.observation_space.parse(state)[0]
                self.best_num_atoms_eval = num_atoms

                if self.relax_best and num_atoms == self.n_atoms_target:
                    calc = XTBOptimizer(method='GFN2-xTB', energy_unit=self.energy_unit)
                    opt_info = calc.optimize_atoms(self.best_molecule_eval)
                    if opt_info['energy_after'] < self.best_eval_energy_relax:
                        self.best_eval_energy_relax = opt_info['energy_after']
                        self.candidate_list.append(
                            MolCandidate(
                                self.best_molecule_eval.copy(),
                                self.best_eval_energy,
                                self.best_eval_reward,
                                opt_info['energy_after'],
                                self.num_episodes_rollout
                            )
                        )

        elif name == 'train':
            self.RL_total_rewards.append(total_reward)
            if energy is not np.nan:
                self.RL_energy.append(energy)
            if total_reward > self.RL_best_reward:
                self.best_count += 1
                self.RL_best_reward = total_reward
                self.RL_best_energy = energy

                self.best_state = state
                self.RL_best_molecule = self.observation_space.parse(state)[0]
                self.best_num_atoms = num_atoms

        else:
            raise ValueError('name must be either eval or train')

        return None


    def update_and_log_data(self, total_num_iter: int) -> None:

        num_episodes_rollout = self.num_episodes_rollout
        self.total_episodes_train += num_episodes_rollout
        self.total_episodes_train_vec.append(self.total_episodes_train)
        self.rollout_iterations = len(self.total_episodes_train_vec)

        # Append mean and std values performance arrays
        self.RL_total_rewards_avg.append(np.mean(self.RL_total_rewards[-num_episodes_rollout:]))
        self.RL_total_rewards_std.append(np.std(self.RL_total_rewards[-num_episodes_rollout:]))

        self.RL_energy_avg.append(np.mean(self.RL_energy[-num_episodes_rollout:]))
        self.RL_energy_std.append(np.std(self.RL_energy[-num_episodes_rollout:]))

        self.RL_best_energy_vec.append(self.RL_best_energy)
        

        #################################################################
        #####################   Plot termination info      ##############
        #################################################################

        # Plot termination info (https://python-graph-gallery.com/250-basic-stacked-area-chart/)

        info_list = self.RL_info[-self.num_episodes_rollout:]
        termination_types = ['full_formula', 'invalid_action', 'stop_token']
        t_count = [info_list.count(t_type) for t_type in termination_types]
        t_count = np.array(t_count, dtype=float) / sum(t_count)
        for i in range(0, len(termination_types)):
            self.RL_info_count[i].append(t_count[i])
        latest_info_count = list(map(list, zip(*self.RL_info_count)))[-1]

        if self.wandb_run is not None:
            try:
                self.wandb_run.log(
                    {
                        # Iteration info
                        "total_num_steps": total_num_iter,
                        "Episodes": self.total_episodes_train,
                        "Rollouts": self.rollout_iterations,
                        "Molecules built": self.mols_built,

                        # Performance
                        "Energy_avg": self.RL_energy_avg[-1],
                        "Energy_std": self.RL_energy_std[-1],
                        "Best_energy": self.RL_best_energy,
                        "Best_excess_energy": self.RL_best_energy-self.benchmark_energy,
                        "Best_excess_energy_relax_eval": self.best_eval_energy_relax-self.benchmark_energy,

                        "Best_reward": self.RL_best_reward,
                        "Reward_avg": self.RL_total_rewards_avg[-1],
                        "Reward_std": self.RL_total_rewards_std[-1],

                        # Benchmarks
                        "Benchmark energy": self.benchmark_energy,
                        # "Benchmark return": self.benchmark_return, # TODO: implement this. for now we can do with energy comparisons

                        # Termnation info
                        "Success_ratio": latest_info_count[0],
                        "invalid_ratio": latest_info_count[1],
                        "stop_token_ratio": latest_info_count[2],

                        # Eval rollouts
                        #"Greedy return": self.eval_return,
                        #"Return deficit": self.GT_best_return-self.eval_return,

                        #"Greedy total cost": self.greedy_agent_total_cost,
                        #"Excess cost": self.greedy_agent_total_cost-self.GT_best_total_cost
                    }
                )
            except Exception as e:
                print(f"An error occurred with BIG wandb log: {e}")


        # pkl dump self.mol_candidates to self.output_dir
        if self.candidate_list:
            with open(os.path.join(self.output_dir, 'mol_candidates.pkl'), 'wb') as f:
                pickle.dump(self.candidate_list, f)

        # Reset counter
        self.num_episodes_rollout = 0


class PretrainLogger():
    def __init__(self, env: gym.Env, output_dir: str, save_to_wandb: bool = False):
        pass

    def log_pretrain_eval(self, eval_results: dict, finetune_formula: str):
        pass



class MultibagLogger(Logger):
    def __init__(self, cf: dict, model: AbstractActorCritic) -> None:
        super().__init__(cf, model)

        self.num_episodes_rollout = 0

    def save_episode_RL(self, state: ObservationType, total_reward: float, info: dict, name: str) -> None:
        self.num_episodes_rollout += 1
        if 'new_rewards' in info:
            info['new_rewards'] = {f'{k}_{name}': v for k, v in info['new_rewards'].items()}
        if 'mol_info' in info:
            info['mol_info'].pop('mol')
        if self.wandb_run is not None:
            self.wandb_run.log(info)
    
    def log_after_rollout(self, info: dict, reward_terms: List[Dict[str, float]], name: str) -> None:
        """ Log the reward terms and other info after full rollout, i.e. many episodes. """
        # List of dicts to dict of lists
        reward_terms = {k: np.mean([d[k] for d in reward_terms]) for k in reward_terms[0]}
        #reward_terms = {k: np.mean(v) for k, v in reward_terms.items()}
        #reward_terms = {f'{k}_{name}': v for k, v in reward_terms.items()}
        info.update(reward_terms)

        #if self.wandb_run is not None:
        #    self.wandb_run.log(info)

    def log_gaussian_stds(self, ac: AbstractActorCritic, total_num_iter: int) -> None:
        if self.wandb_run is None:
            return None
        if isinstance(ac, PainnInternalMultiModal):
            distance_std = np.exp(ac.distance_log_stds.detach().cpu().numpy()).mean()
            angle_std = np.exp(ac.angle_log_stds.detach().cpu().numpy()).mean()
            dihedral_std = np.exp(ac.dihedral_log_stds.detach().cpu().numpy()).mean()
            self.wandb_run.log({'total_num_steps': total_num_iter,
                                'distance_std': distance_std,
                                'angle_std': angle_std,
                                'dihedral_std': dihedral_std})
        elif isinstance(ac, PainnAC):
            stds = ac.log_stds.exp().detach().cpu().numpy()
            if isinstance(ac, NewAC):
                self.wandb_run.log({'total_num_steps': total_num_iter, 
                                    'distance_std': stds[0],
                                    'euclidean_std': stds[1]})
            else:
                self.wandb_run.log({'total_num_steps': total_num_iter, 
                                    'distance_std': stds[0],
                                    'angle_std': stds[1],
                                    'dihedral_std': stds[2]})
        else:
            return None
