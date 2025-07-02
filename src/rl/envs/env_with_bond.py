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
from src.rl.spaces_2d import ActionSpace, ObservationSpace2d, ActionType, ObservationType, FormulaType
from src.tools.util import remove_atom_from_formula, get_formula_size, zs_to_formula


class AbstractMolecularEnvironment2d(gym.Env, abc.ABC):

    def __init__(
        self,
        reward: InteractionReward,
        observation_space: ObservationSpace2d,
        action_space: ActionSpace,
        min_atomic_distance=0.6,  # Angstrom
        max_solo_distance=2.0,  # Angstrom
        min_reward=-0.6,  # Hartree
        worker_id=None,
        seed=0,
        terminal_reward_only=True
    ):
        self.reward = reward
        self.observation_space = observation_space
        self.action_space = action_space

        self.random_state = np.random.RandomState(seed=seed)

        self.min_atomic_distance = min_atomic_distance
        self.max_solo_distance = max_solo_distance
        self.min_reward = min_reward

        self.current_atoms = Atoms()
        self.current_formula: FormulaType = tuple()
        self.worker_id = worker_id
        self.terminal_reward_only = terminal_reward_only


    @abc.abstractmethod
    def reset(self) -> ObservationType:
        raise NotImplementedError

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        canvas_item, atom_connectivity = action
        atomic_number_index, position = canvas_item
        atomic_number = self.action_space.zs[atomic_number_index]
        done = atomic_number == 0

        if done:
            return self.observation_space.build(self.current_atoms, self.current_formula, self.connectivity), 0.0, done, {'termination_info': 'stop_token'}

        new_atom = self.action_space.canvas_item_space.to_atom(canvas_item)
        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom):
            return (
                self.observation_space.build(self.current_atoms, self.current_formula),
                self.min_reward,
                True,
                {'termination_info': 'invalid_action'},
            )

        if self.terminal_reward_only:
            if self._is_terminal(lag=1):
                reward, info = self._calculate_reward(new_atom)
                info['termination_info'] = 'full_formula'
                # logging.info(f'--------------------------------------------------------------------------- Final reward:  {reward}')
            else:
                reward, done, info = 0.1, False, {}
        else:
            reward, info = self._calculate_reward(new_atom)
            print(f'reward:  {reward}')
            if reward < self.min_reward:
                done = True
                print(f'TERMINATING DUE TO MIN REWARD:  {reward}')
                reward = self.min_reward


        self.current_atoms.append(new_atom)
        self.current_formula = remove_atom_from_formula(self.current_formula, atomic_number)

        # Check if state is terminal
        if self._is_terminal(lag=0):
            done = True
        
        return self.observation_space.build(self.current_atoms, self.current_formula, self.connectivity), reward, done, info

    def _is_terminal(self, lag: int = 1) -> bool:
        return len(self.current_atoms) + lag == self.observation_space.canvas_space.size or get_formula_size(
            self.current_formula) == lag

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False

        return self._all_covered(current_atoms, new_atom)

    def _is_too_close(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Check distances between new and old atoms
        for existing_atom in existing_atoms:
            if np.linalg.norm(existing_atom.position - new_atom.position) < self.min_atomic_distance:
                logging.debug('Atoms are too close')
                return True

        return False

    def _calculate_reward(self, new_atom: Atom) -> Tuple[float, dict]:
        reward, info = self.reward.calculate(self.current_atoms, new_atom, self.worker_id, self.terminal_reward_only)
        if reward == -42.0: # then it has failed
            return self.min_reward, info
        return reward, info

    def _all_covered(self, existing_atoms: Atoms, new_atom: Atom) -> bool:
        # Ensure that certain atoms are not too far away from the nearest heavy atom to avoid H2, F2,... formation
        candidates = ['H', 'F', 'Cl', 'Br']
        if len(existing_atoms) == 0 or new_atom.symbol not in candidates:
            return True

        for existing_atom in existing_atoms:
            if existing_atom.symbol in candidates:
                continue

            distance = np.linalg.norm(existing_atom.position - new_atom.position)
            if distance < self.max_solo_distance:
                return True

        logging.debug('There is a single atom floating around')
        return False

    def render(self, mode='human'):
        pass

    def seed(self, seed=None) -> int:
        seed = seed or np.random.randint(int(1e5))
        self.random_state = np.random.RandomState(seed)
        return seed



class HeavyFirst2d(AbstractMolecularEnvironment2d):
    def __init__(self, formulas: List[FormulaType], benchmark_energy: List[float] = [None], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.formulas = formulas
        self.formula_cycle = itertools.cycle(self.formulas)

        self.benchmark_energies = benchmark_energy
        self.benchmark_energy_cycle = itertools.cycle(self.benchmark_energies)

        self.formula_counter = 0
        self.reshuffle_formula_list()

        self.obs_reset = self.reset()

    def reset(self) -> ObservationType:
        if self.formula_counter == len(self.formulas):
            self.reshuffle_formula_list()

        self.current_atoms = Atoms()
        self.current_formula = next(self.formula_cycle)
        max_atoms = self.observation_space.canvas_space.size
        self.connectivity = np.zeros((max_atoms, max_atoms), dtype=np.int8)

        self.benchmark_energy = next(self.benchmark_energy_cycle)
        self.reward.reset_old_energies(self.worker_id)

        # Take a step to add the heaviest atom on the canvas
        heaviest = max([z for (z, _) in self.current_formula ])
        heaviest_number_index = self.action_space.zs.index(heaviest)
        obs, _, _, _ = self.step(
            action=(
                (heaviest_number_index, (0, 0, 0)),
                self.connectivity[0],
            )
        )

        return obs

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False
        return True
    
    def reshuffle_formula_list(self):
        self.formula_counter = 0
        random_permutation = np.random.permutation(len(self.formulas))
        self.formulas = [self.formulas[i] for i in random_permutation]
        self.formula_cycle = itertools.cycle(self.formulas)

        if self.benchmark_energies[0] is not None:
            self.benchmark_energies = [self.benchmark_energies[i] for i in random_permutation]
            self.benchmark_energy_cycle = itertools.cycle(self.benchmark_energies)
