from typing import Tuple, List

import gym
import numpy as np
from ase import Atoms

from src.rl.spaces import (
    CanvasItemType, 
    CanvasItemSpace, 
    CanvasType,
    CanvasSpace, 
    ActionType,
    ActionSpace,
    ObservationSpace
)


NumAtomsToGoType = int  # Integer indicating the number of atoms to go
ObservationType = Tuple[CanvasType, NumAtomsToGoType]
# TODO: # Introduce a "MaskType = np.ndarray"  # A 1D numpy array with shape (max_atoms)


class ObservationSpace(gym.spaces.Tuple):
    def __init__(self, canvas_size: int, zs: List[int]):
        self.zs = zs
        self.canvas_space = CanvasSpace(size=canvas_size, zs=zs)
        self.num_atoms_to_go_space = gym.spaces.Discrete(n=canvas_size)

        super().__init__((self.canvas_space, self.num_atoms_to_go_space))

    def build(self, atoms: Atoms, n_atoms_to_go: int) -> ObservationType:
        return self.canvas_space.from_atoms(atoms), n_atoms_to_go

    def parse(self, observation: ObservationType) -> Tuple[Atoms, int]:
        return self.canvas_space.to_atoms(observation[0]), observation[1]
