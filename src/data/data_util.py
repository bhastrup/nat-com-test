
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from collections import Counter
from ase.data import chemical_symbols
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as RDKitBondType



############## Basic bag manipulation functions ################
def elements_to_symbols(elements: set[int]) -> List[str]:
    return [chemical_symbols[atomic_number] for atomic_number in elements]

def syms_to_count_dict(atom_syms: List[str], symbols_sorted: List[str]) -> Dict[str, int]:
    counts = dict(Counter(atom_syms))
    return {k: counts[k] for k in symbols_sorted}

def sym_count_to_bag_repr(counts_sort: Dict[str, int]):
    return ''.join([f'{k}' if v == 1 else f'{k}{v}' for k, v in counts_sort.items()])


#############################################################################
def get_benchmark_energies_from_df(df: pd.DataFrame) -> Dict[str, float]:
    return df.groupby('bag_repr')['energy_GFN2'].mean().to_dict()


# Example usage:
# energies_in_eV = convert_energy_units(energies_in_hartree, 'hartree', 'eV')


##################### Connectivity matrix functionality #####################
bondtype_to_order = {
    RDKitBondType.SINGLE: 1,
    RDKitBondType.DOUBLE: 2,
    RDKitBondType.TRIPLE: 3,
    RDKitBondType.AROMATIC: 4
}

valency = {
    'H': (1,),
    'C': (4,),
    'N': (3,),
    'O': (2,),
    'S': (2, 6)
}

def valence_is_good(atom, valency):
    return np.any([atom.GetExplicitValence() == v for v in valency[atom.GetSymbol()]]) 


def get_con_mat(mol: Chem.Mol) -> np.ndarray:
    """ Returns a connectivity matrix of the molecule using the bondtype_to_order."""
    n_atoms = mol.GetNumAtoms()
    con_mat = np.zeros((n_atoms, n_atoms), dtype=np.int8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        con_mat[i, j] = bondtype_to_order[bond.GetBondType()]
        con_mat[j, i] = bondtype_to_order[bond.GetBondType()]

    return con_mat


def get_con_sparse(mol: Chem.Mol) -> dict:
    """ Returns a dictionary of the molecule connectivity matrix in sparse format."""
    
    bond_dict = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_dict[(i, j)] = bondtype_to_order[bond.GetBondType()]
        
    return bond_dict


def get_adj_mat(mol: Chem.Mol) -> np.ndarray:
    """ Returns an adjacency matrix of the molecule."""
    n_atoms = mol.GetNumAtoms()
    adj_mat = np.zeros((n_atoms, n_atoms), dtype=np.int8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    return adj_mat


def get_con_upper_triangular(mol: Chem.Mol) -> np.ndarray:
    """ 
    Returns the upper triangular part of the connectivity matrix of the molecule.
    Uses np.int8 for bond orders to optimize memory usage.
    """
    n_atoms = mol.GetNumAtoms()

    # Calculate the length of the upper triangular array (excluding the diagonal)
    upper_triangular_length = n_atoms * (n_atoms - 1) // 2

    # Initialize the upper triangular array
    upper_triangular = np.zeros(upper_triangular_length, dtype=np.int8)

    # Function to map matrix indices to the upper triangular array index
    def matrix_to_upper_triangular_idx(i, j):
        if i < j:
            return i * n_atoms - (i * (i + 1) // 2) + j - i - 1
        else:
            return j * n_atoms - (j * (j + 1) // 2) + i - j - 1

    # Fill in the upper triangular array
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_order = bondtype_to_order[bond.GetBondType()]
        idx = matrix_to_upper_triangular_idx(i, j)
        upper_triangular[idx] = bond_order

    return upper_triangular


def unpack_upper_triangular_to_full(upper_triangular: np.ndarray, n_atoms: int) -> np.ndarray:
    """
    Converts an upper triangular array back into a full 2D connectivity matrix.
    
    :param upper_triangular: 1D np.ndarray representing the upper triangular part of the matrix.
    :param n_atoms: The number of atoms in the molecule.
    :return: 2D np.ndarray representing the full connectivity matrix.
    """
    # Initialize the full matrix
    con_mat = np.zeros((n_atoms, n_atoms), dtype=np.int8)

    # Function to map upper triangular array index back to matrix indices
    def upper_triangular_to_matrix_idx(idx):
        for i in range(n_atoms):
            if idx < n_atoms - i - 1:
                return i, idx + i + 1
            idx -= n_atoms - i - 1

    # Fill in the full matrix
    for idx, bond_order in enumerate(upper_triangular):
        i, j = upper_triangular_to_matrix_idx(idx)
        con_mat[i, j] = con_mat[j, i] = bond_order

    return con_mat
