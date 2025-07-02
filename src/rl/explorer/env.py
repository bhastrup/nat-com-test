import abc
import itertools
import logging
from typing import Tuple

import gym
import numpy as np
from ase import Atoms, Atom

from src.rl.reward import InteractionReward
from src.rl.envs.environment import AbstractMolecularEnvironment

from src.rl.explorer.spaces import (
    ActionSpace, 
    ObservationSpace, 
    ActionType, 
    ObservationType
)


class AbstractEnvAtomsToGo(AbstractMolecularEnvironment):

    def apply_explorer(self, action: ActionType, inject_noise: bool = False) -> None:
        atomic_number_index, _ = action
        atomic_number = self.action_space.zs[atomic_number_index]
        done = atomic_number == 0
        if done:
            return (self.observation_space.build(self.current_atoms, self.current_num_atoms_to_go), 
                    0.0, done, {'termination_info': 'stop_token'})

        # Accept the action
        new_atom = self.action_space.to_atom(action)
        self.current_atoms.append(new_atom)
        if inject_noise:
            noise = np.random.normal(0, 0.1, size=self.current_atoms.get_positions().shape)
            self.current_atoms.set_positions(self.current_atoms.get_positions() + noise)

        self.current_num_atoms_to_go -= 1
    
    def is_valid_action(self, atoms: Atoms) -> bool:
        if self.molecule_has_blown_up(atoms):
            return False
        else:
            return True

    def apply_corrector(self, updated_atoms: Atoms) -> None:
        if not self.is_valid_action(atoms=updated_atoms):
            return (
                self.observation_space.build(updated_atoms, self.current_num_atoms_to_go),
                self.min_reward,
                True,
                {'termination_info': 'corrector_blew_up'},
            )
    
        # Update state with Corrector predictions
        self.current_atoms = updated_atoms

    def post_step(self) -> Tuple[ObservationType, float, bool, dict]:
        reward, done, info = self._reward_logic(self._is_terminal(lag=0), self.use_intermediate_rewards)
        return self.observation_space.build(self.current_atoms, self.current_num_atoms_to_go), reward, done, info

    def _is_terminal(self, lag: int = 1) -> bool:
        return len(self.current_atoms) + lag == self.observation_space.canvas_space.size or self.current_num_atoms_to_go == lag

    def _reward_logic(
            self, 
            terminal: bool, 
            use_intermediate_rewards: bool = False
    ) -> Tuple[float, bool, dict]:

        done = True if terminal else False
        if not use_intermediate_rewards and not terminal:
            return self.constant_reward, done, {}
        
        new_atom = self.current_atoms[-1] # Split atoms to conform with calculate() args
        all_but_new = self.current_atoms[:-1]
        reward, info = self.reward.calculate(all_but_new, new_atom, terminal, self.worker_id)
        if terminal:
            info['termination_info'] = 'full_formula'
        
        if reward is None:
            return self.min_reward, True, info
        
        return reward, done, info

    def molecule_has_blown_up(self, atoms: Atoms, threshold: float = 5.0) -> bool:
        """Checks if the molecule has 'blown up' by determining if the smallest interatomic 
        distance is too large, indicating disconnected atoms."""
        pos = np.array(atoms.get_positions())
        distances = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2)
        np.fill_diagonal(distances, np.inf)
        smallest_dist = np.min(distances)
        return smallest_dist > threshold




class NumAtomsToGoRangeEnv(AbstractEnvAtomsToGo):
    def __init__(self, num_atoms_to_go_interval: Tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Change to Empirical Distribution
        self.num_atoms_to_go_interval = num_atoms_to_go_interval
        self.num_atoms_range = np.arange(
            num_atoms_to_go_interval[0], 
            num_atoms_to_go_interval[1] + 1
        )

        self.reshuffle_num_atoms()
        self.obs_reset = self.reset()

    def reshuffle_num_atoms(self):
        self.num_atoms_counter = 0
        permuted_range = np.random.permutation(len(self.num_atoms_range))
        self.num_atoms_iterable = [self.num_atoms_range[i] for i in permuted_range]
        self.num_atoms_cycle = itertools.cycle(self.num_atoms_iterable)

    def reset(self) -> ObservationType:
        self.reward.reset_old_energies(self.worker_id)
        if self.num_atoms_counter == len(self.num_atoms_range):
            self.reshuffle_num_atoms()

        self.current_atoms = Atoms()
        self.current_num_atoms_to_go = next(self.num_atoms_cycle)
        return self.observation_space.build(self.current_atoms, self.current_num_atoms_to_go)


class NumAtomsToGoFixedEnv(AbstractEnvAtomsToGo):
    def __init__(self, num_atoms_to_go: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_atoms_to_go = num_atoms_to_go

    def reset(self) -> ObservationType:
        self.reward.reset_old_energies(self.worker_id)
        self.current_atoms = Atoms()
        self.current_num_atoms_to_go = self.num_atoms_to_go
        return self.observation_space.build(self.current_atoms, self.current_num_atoms_to_go)
