from typing import Tuple, List

import gym
import numpy as np
from ase import Atoms


NULL_SYMBOL = 'X'

from .spaces import (
    CanvasItemType, 
    CanvasType, 
    BagType, 
    FormulaType,
    CanvasItemSpace, 
    CanvasSpace, 
    BagSpace
)


# Alternatively: https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html (BondType) 
# or: BondType = Tuple[int, ...] # length b+1, with b=#bond types.

ConnectivityType = np.ndarray  # A 2D numpy array with shape (max_atoms, max_atoms)
ObservationType = Tuple[CanvasType, BagType, ConnectivityType]


AtomConnectivityType = np.ndarray  # A 1D numpy array with length max_atoms
ActionType = Tuple[CanvasItemType, AtomConnectivityType]


class ActionSpace(gym.spaces.Tuple):
    def __init__(self, zs: List[int], max_atoms: int):
        assert 0 in zs, '0 has to be in the list of atomic numbers'

        self.zs = zs
        self.max_atoms = max_atoms

        self.canvas_item_space = CanvasItemSpace(zs=zs)
        self.bond_space = gym.spaces.Box(low=0, high=3, shape=(max_atoms,), dtype=np.int8)

        super().__init__((self.canvas_item_space, self.bond_space))



class ConnectivitySpace(gym.spaces.Box):
    def __init__(self, max_atoms: int):
        super().__init__(low=0, high=3, shape=(max_atoms, max_atoms), dtype=np.int8)
        self.max_atoms = max_atoms

    def to_connectivity(self, connectivity_matrix: ConnectivityType):
        # Your logic to convert the connectivity matrix to the desired format
        # For example, return as is or apply some transformation if needed
        return connectivity_matrix

    def from_connectivity(self, connectivity_data) -> ConnectivityType:
        # Your logic to convert external connectivity data into the matrix format
        # This might involve padding for smaller molecules, etc.
        return connectivity_data



class ObservationSpace2d(gym.spaces.Tuple):
    def __init__(self, canvas_size: int, zs: List[int]):
        self.zs = zs
        self.canvas_space = CanvasSpace(size=canvas_size, zs=zs)
        self.bag_space = BagSpace(zs=zs)
        self.connectivity_space = ConnectivitySpace(max_atoms=canvas_size)
        super().__init__((self.canvas_space, self.bag_space, self.connectivity_space))

    def build(self, atoms: Atoms, formula: FormulaType, connectivity: ConnectivityType) -> ObservationType:
        return self.canvas_space.from_atoms(atoms), self.bag_space.from_formula(formula), \
            self.connectivity_space.from_connectivity(connectivity)

    def parse(self, observation: ObservationType) -> Tuple[Atoms, FormulaType]:
        return self.canvas_space.to_atoms(observation[0]), self.bag_space.to_formula(observation[1]), \
            self.connectivity_space.to_connectivity(observation[2])
