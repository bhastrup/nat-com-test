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

from src.rl.envs.env_with_bond import AbstractMolecularEnvironment2d



import random
from collections import deque


import pandas as pd
from src.pretraining.action_decom import recenter, gaussian_perturbation, decompose_pos



class ObservationBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.current_age = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, obs):
        """Add an observation with an age to the buffer."""
        self.buffer.append((obs, self.current_age))
        self.current_age += 1

    def sample(self, batch_size):
        """Sample observations with priority given to older observations, without replacement."""
        if batch_size > len(self.buffer):
            raise ValueError("Sample size greater than number of elements in buffer")

        ages = [age for (_, age) in self.buffer]
        probs = [age / sum(ages) for age in ages]
        sampled_indices = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False, p=probs)
    
        # Extract the data from the buffer
        samples = [self.buffer[i][0] for i in sampled_indices]  

        # Remove sampled elements
        for i in sorted(sampled_indices, reverse=True):
            del self.buffer[i]

        return samples


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

        
class CanvasGenerator:
    """ Holds a dataset of molecules. When requested, it decomposes a molecule and returns
        a partial molecule (canvas), that the RL agent can complete. The skill level starts out 
        low but should be increased during training. It reflects the number of atoms that are
        yet to be placed on the canvas in order to complete the molecule. """

    def __init__(self, df: pd.DataFrame, config: dict, capacity: int = 5000):
        self.df = df
        self.config = config
        self.skill = 1

        assert capacity > 200
        self.capacity = capacity
        self.fill_thres = self.capacity - 100
        self.buffer_min_size = capacity // 4
        self.buffer = ObservationBuffer(capacity=capacity)
        self.scheme = 'random'

    def __len__(self):
        return len(self.df)

    def get_new_canvas(self) -> Tuple[Atoms, FormulaType]:
        if len(self.buffer) < self.fill_thres:
            self.fill_buffer()
        while len(self.buffer) < self.buffer_min_size:
            self.fill_buffer()
            print(f"buffer size: {len(self.buffer)}")

        return self.buffer.sample(1)[0]


    def fill_buffer(self) -> None:
        mol = self.sample_molecule()
        canvas_list = self.create_partial_molecules(mol=mol)
        self.push_canvases_to_buffer(canvas_list=canvas_list)


    def sample_molecule(self) -> dict:
        """ Pick a random molecule from the dataset """
        # TODO: Should we do this with replacement or as an epoch?
        index = random.randint(0, len(self.df)-1)
        id = self.df.index[index]
        return self.df.loc[id].to_dict().copy()


    def push_canvases_to_buffer(self, canvas_list: List[Tuple[Atoms, FormulaType]]) -> None:
        """ Add the canvases to the buffer. We create duplicates of the canvases in proportion the 
            number of atoms to place, to avoid populating the buffer with only 1-atom tasks. """

        if len(canvas_list) == 0:
            return

        multiplier = 1 # Simply to speed up buffer refilling. Could easily be a lot higher.
        in_prop_to_size = False # If true, the number of duplicates is proportional to the number of atoms to place

        for canvas in canvas_list:
            for _ in range(multiplier * (len(canvas) if in_prop_to_size else 1)):
                self.buffer.push(canvas)


    def create_partial_molecules(self, mol: dict) -> List[Tuple[Atoms, FormulaType]]:
        # no_hydrogen_focus = config['no_hydrogen_focus']

        pos, elements, symbols, formula, con_mat = mol['pos'], mol['atomic_nums'], mol['atomic_symbols'], mol['formulas'], mol['con_mat']

        # pos = gaussian_perturbation(pos, sigma=None)
        pos = recenter(pos, elements, formula, self.config['mol_dataset'], heavy_first=False)


        # Decompose the molecule
        p_random = self.config['decom_p_random']
        scheme = np.random.choice(['random', 'standard', 'ring_reconstruction'], p=[p_random, 1-p_random, 0.0])

        if scheme == 'random':
            sorted_indices, num_atoms_to_place = self.decompose_random(elements, pos)
        elif scheme == 'standard':
            sorted_indices, num_atoms_to_place = self.decompose_from_center(elements, pos)
        elif scheme == 'ring_reconstruction':
            sorted_indices = self.decompose_ring_reconstruction(mol)

        if sorted_indices is None:
            return []

        return self.construct_observation_tuples2d(pos, elements, con_mat, sorted_indices, num_atoms_to_place)


    def decompose_from_center(self, elements: List[int], pos: np.ndarray) -> Tuple[np.ndarray, int]:
        """ Decompose the molecule from the center."""

        decom_method = self.config['decom_method']
        cutoff = self.config['decom_cutoff']
        shuffle = self.config['decom_shuffle']
        mega_shuffle = self.config['decom_mega_shuffle']
        hydrogen_delay = self.config['hydrogen_delay']
        
        # Can be any length
        nmin = 1
        nmax = len(elements)
        num_atoms_to_place = np.random.choice(range(nmin, nmax+1))

        sorted_indices = None
        iter = 0
        while type(sorted_indices) != np.ndarray:
            iter += 1
            if iter > 10:
                print("Couldn't find a valid decomposition")
                return (None, None)
            sorted_indices = decompose_pos(elements, pos, decom_method=decom_method, cutoff=cutoff, 
                                           shuffle=shuffle, mega_shuffle=mega_shuffle, 
                                           hydrogen_delay=hydrogen_delay)
            
    
        return sorted_indices, num_atoms_to_place


    def decompose_random(self, elements: List[int], pos: np.ndarray) -> Tuple[np.ndarray, int]:
        """ Decompose the molecule randomly.
            We have to makes sure the core (canvas) is connected.
            Otherwise GNN message passing cannot perceive the full structure. """

        # Can only be 1, 2 or 3. Otherwise it gets too hard to learn.
        nmin = 1
        nmax = 1
        num_atoms_to_place = np.random.choice(range(nmin, nmax+1))


        elements = np.array(elements)
        num_core = len(elements) - num_atoms_to_place

        # print(f"core: {num_core} / total: {len(elements)}")


        def core_contains_heavy(sorted_indices):
            """" Core canvas cannot be filled solely with hydrogen """
            if num_core == 0:
                return True
            return max(elements[sorted_indices[:num_core]]) > 1
        
        def core_is_connected(sorted_indices):
            """ Checks that the entire molecule is connected (in terms of GNN cutoff distance) """
            if num_core == 0:
                return True
            pos_new = pos[sorted_indices[:num_core]]
            mutual_distances = np.linalg.norm(pos_new[:, None, :] - pos_new[None, :, :], axis=-1)
            all_connected = False
            while not all_connected:
                connected = mutual_distances[0] < self.config['cutoff']
                while not all_connected:
                    old_connected = connected.copy()
                    connected = np.logical_or(
                        old_connected, 
                        np.any(mutual_distances[connected] < self.config['cutoff'], 
                    axis=0))
                    if sum(connected) == len(connected):
                        return True
                    if np.all(old_connected == connected):
                        return False


        # Find the heaviest atom
        #center_element = max(elements)
        #center_element = elements.index(center_element)


        sorted_indices = np.random.permutation(len(elements))
        while not (core_contains_heavy(sorted_indices) and core_is_connected(sorted_indices)):
            # print(f"didn't pass core_contains_heavy or core_is_connected. num_core: {num_core}, len(elements): {len(elements)}")
            # atoms_object = Atoms(numbers=elements, positions=pos)
            # from ase.visualize import view
            # view(atoms_object)

            # failed_atoms = Atoms(numbers=elements[sorted_indices[:num_core]], positions=pos[sorted_indices[:num_core]])
            # view(failed_atoms)

            sorted_indices = np.random.permutation(len(elements))

        return sorted_indices, num_atoms_to_place


    def decompose_ring_reconstruction(self, mol: dict) -> List[Tuple[Atoms, FormulaType]]:
        """ Contrary to 'decompose_pos', here we make sure that the leftout atoms were
            part of a ring which the agent should now attempt to reconstruct.
        """
        # Probably we should preprocess the entire dataset and save a boolen array,
        # indicating which atoms are part of a ring.

        # 1. Select a ring member atom at random

        # 2. Remove the atom from the molecule

        # 3. Remove a few others also depending on skill?

        pass # just use scheme == 'standard' for now

        
    def num_atoms_to_place(self, mol: dict) -> int:
        # TODO: Replace with some sampling (like a Gamma distribution) 
        #       to bring some variation to the number of atoms to place

        n_atoms_min = 1 # Minimum number of atoms to place
        n_atoms_max = len(mol['atomic_nums'])

        # Option a) Sample n_atoms in range between 1 and all_atoms
        n_atoms = np.random.choice(range(n_atoms_min, n_atoms_max+1)) # use +1 if we wish to play on entirely empty canvas
        # The elegant one.

        # Option b) Just use skill directly
        # n_atoms = self.skill
        # Arghya insists on this one

        # Option c) Sample number between n_atoms_min and skill
        # n_atoms = np.random.choice(range(n_atoms_min, self.skill+1))
        # If skill should be used, it should be like this.

        return clamp(n_atoms, n_atoms_min, n_atoms_max)

    def construct_observation_tuples2d(
        self,
        pos,
        elements,
        con_mat,
        sorted_indices,
        num_atoms_to_place: int = 1
    ) -> List[Tuple[Atoms, FormulaType]]:
        """ Construct observation tuples from decomposed molecule."""

        num_core = len(elements) - num_atoms_to_place

        # print(f"sorted indices: {sorted_indices}")
        # print(f"num core: {num_core}")
        # print(f"num atoms to place: {num_atoms_to_place}")

        pos = pos[sorted_indices]
        elements = np.array(elements)[sorted_indices]
        con_mat = con_mat[sorted_indices][:, sorted_indices]

        obs_tuples = []
        for i in range(num_atoms_to_place):
            # Expand the canvas with one atom at a time
            num_atoms_new = num_core + i
            pos_new = pos[:num_atoms_new]
            zs_new = elements[:num_atoms_new]
            canvas_atoms = Atoms(numbers=zs_new, positions=pos_new)
            canvas_connectivity = con_mat[:num_atoms_new][:, :num_atoms_new]


            # Similarly reduce the bag of atoms to place
            bag_formula = zs_to_formula(elements[sorted_indices[num_atoms_new:]])

            obs_tuples.append((canvas_atoms, bag_formula, canvas_connectivity))
        
        # print(f"obs tuples: {obs_tuples}")

        return obs_tuples




class PartialCanvasEnv2d(AbstractMolecularEnvironment2d):
    def __init__(self, molecule_df: pd.DataFrame, decom_params: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # benchmark_energy: List[float] = [None]
        # self.eval = eval
        self.canvas_generator = CanvasGenerator(df=molecule_df, config=decom_params)
        self.obs_reset = self.reset()

    def reset(self) -> ObservationType:
        # TODO: When requesting a new canvas, we currently recenter according to the heavist atom.
        #       This can be relaxed such that we sample 

        self.current_atoms, self.current_formula, con_mat = self.canvas_generator.get_new_canvas()
        self.connectivity = np.zeros((self.observation_space.canvas_space.size, self.observation_space.canvas_space.size), dtype=np.int8)
        self.connectivity[:len(self.current_atoms), :len(self.current_atoms)] = con_mat

        return self.observation_space.build(self.current_atoms, self.current_formula, self.connectivity)

    def _is_valid_action(self, current_atoms: Atoms, new_atom: Atom, con_mat: np.ndarray) -> bool:
        if self._is_too_close(current_atoms, new_atom):
            return False
        
        elements = current_atoms.get_atomic_numbers()
        new_elements = np.append(elements, new_atom.number)

        print(f"con_mat: {con_mat}")
        print(f"new_elements: {new_elements}")
        exit()
        if self._has_too_many_bonds(con_mat, new_atom):
            return False
        return True

    def step(self, action: ActionType) -> Tuple[ObservationType, float, bool, dict]:
        atomic_number_index, position, con_mat = action
        atomic_number = self.action_space.zs[atomic_number_index]
        done = atomic_number == 0

        if done:
            return self.observation_space.build(self.current_atoms, self.current_formula, self.connectivity), 0.0, done, {'termination_info': 'stop_token'}

        new_atom = self.action_space.to_atom(action)
        if not self._is_valid_action(current_atoms=self.current_atoms, new_atom=new_atom, con_mat=con_mat):
            return (
                self.observation_space.build(self.current_atoms, self.current_formula, self.connectivity),
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


    def _has_too_many_bonds(con_mat: np.ndarray, elements: List[int]) -> bool:
        """ Sums up the number of bonds for each atom and compares to the valency. """
        assert con_mat.shape[0] == len(elements)

        print(f"con_mat: {con_mat}")
        exit()

        valency = {
            'H': 1,
            'C': 4,
            'N': 3,
            'O': 2,
            'S': 2
        }

        # print(f"con_mat: {con_mat}")
        # print(f"elements: {elements}")

        bonds = np.sum(con_mat, axis=0)
        symbols = [ase.data.chemical_symbols[z] for z in elements]
        if not np.all([bonds[i] <= valency[symbols[i]] for i in range(len(elements))]):
            return False


valency = {
    'H': (1,),
    'C': (4,),
    'N': (3,),
    'O': (2,),
    'S': (2, 6)
}


def correct_final_valence_rdkit(atom, valency):
    """ Uses Chem.Mol so not very efficient. """
    return np.all(np.any([atom.GetExplicitValence() == v for v in valency[atom.GetSymbol()]]))

