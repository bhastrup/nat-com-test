import abc
import itertools
import logging
from typing import Tuple, List, Callable

import ase.data
import gym
import numpy as np
from ase import Atoms, Atom
from scipy.spatial.qhull import ConvexHull, Delaunay

from src.rl.reward import InteractionReward
from src.rl.spaces import ActionSpace, ObservationSpace, ActionType, ObservationType, FormulaType
from src.tools.util import remove_atom_from_formula, get_formula_size, zs_to_formula


from src.rl.envs.environment import AbstractMolecularEnvironment, HeavyFirst



# class HeavyFirst(AbstractMolecularEnvironment):
#     def __init__(self, formulas: List[FormulaType], benchmark_energy: List[float] = [None], *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.formulas = formulas
#         self.formula_cycle = itertools.cycle(self.formulas)

#         self.benchmark_energies = benchmark_energy
#         self.benchmark_energy_cycle = itertools.cycle(self.benchmark_energies)

#         self.formula_counter = 0
#         self.reshuffle_formula_list()

#         self.obs_reset = self.reset()

#     def reset(self) -> ObservationType:
#         if self.formula_counter == len(self.formulas):
#             self.reshuffle_formula_list()

#         self.current_atoms = Atoms()
#         self.current_formula = next(self.formula_cycle)
#         self.benchmark_energy = next(self.benchmark_energy_cycle)
#         self.reward.reset_old_energies(self.worker_id)

#         # Take a step to add the heaviest atom on the canvas
#         heaviest = max([z for (z, _) in self.current_formula ])
#         heaviest_number_index = self.action_space.zs.index(heaviest)
#         obs, _, _, _ = self.step(action=(heaviest_number_index, (0, 0, 0)))

#         return obs


class HeavyFirstNoReward(HeavyFirst):


    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        atomic_number_index, position = action
        atomic_number = self.action_space.zs[atomic_number_index]
        done = atomic_number == 0

        if done:
            return self.observation_space.build(self.current_atoms, self.current_formula), 0.0, done, {'termination_info': 'stop_token'}

        new_atom = self.action_space.to_atom(action)
        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom):
            return (
                self.observation_space.build(self.current_atoms, self.current_formula),
                self.min_reward,
                True,
                {'termination_info': 'invalid_action'},
            )

        self.current_atoms.append(new_atom)
        self.current_formula = remove_atom_from_formula(self.current_formula, atomic_number)

        # Check if state is terminal
        if self._is_terminal(lag=0):
            done = True

        
        reward = 0
        info = {}

        
        return self.observation_space.build(self.current_atoms, self.current_formula), reward, done, info

    def reset(self) -> ObservationType:
        if self.formula_counter == len(self.formulas):
            self.reshuffle_formula_list()

        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)
        self.benchmark_energy = next(self.benchmark_energy_cycle)
        #self.reward.reset_old_energies(self.worker_id)

        # Take a step to add the heaviest atom on the canvas
        heaviest = max([z for (z, _) in self.current_formula ])
        heaviest_number_index = self.action_space.zs.index(heaviest)
        obs, _, _, _ = self.step(action=(heaviest_number_index, (0, 0, 0)))

        return obs
