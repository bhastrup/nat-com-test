

from typing import List
import sys
import warnings
import logging
import multiprocessing
import threading
import torch
import numpy as np
import scipy.spatial
import ase.db
import pandas as pd

try:
    import asap3
except ModuleNotFoundError:
    warnings.warn("Failed to import asap3 module for fast neighborlist")


from src.agents.painn.data_painn import AseNeigborListWrapper, _cell_heights

# Based on https://github.com/peterbjorgensen/DeepDFT/blob/main/dataset.py

##################################################################################
### For ExplorerAgent which uses probe points for next placement decisions #######
##################################################################################

# class TransformAtomsObjectsToGraphXyz:
#     """
#     Transform Atoms() to graph while keeping the xyz positions of the vertices

#     """

#     def __init__(self, cutoff=5.0):
#         self.cutoff = cutoff

#     def __call__(self, atoms):

#         edges, edges_displacement = self.get_edges_simple(atoms)

#         default_type = torch.get_default_dtype()

#         # pylint: disable=E1102
#         graph_data = {
#             "nodes": torch.tensor(atoms.get_atomic_numbers()),
#             "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
#             "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
#             "edges": torch.tensor(edges),
#             "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
#             "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
#             "num_edges": torch.tensor(edges.shape[0])
#         }

#         return graph_data

#     def get_edges_simple(self, atoms):
#         # Compute distance matrix
#         pos = atoms.get_positions()
#         dist_mat = scipy.spatial.distance_matrix(pos, pos)

#         # Build array with edges and edge features (distances)
#         valid_indices_bool = dist_mat < self.cutoff
#         np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
#         edges = np.argwhere(valid_indices_bool)  # num_edges x 2
#         edges_displacement = np.zeros((edges.shape[0], 3))

#         return edges, edges_displacement



# class AtomsToMoleculeGraph:
#     """
#     Convert ase.Atoms() to a graph dictionary. 
#     """

#     def __init__(self, cutoff=5.0):
#         self.cutoff = cutoff


#     def __call__(self, atoms, trial_poses):
#         edges, edges_displacement = self.get_edges_simple(atoms)


#         default_type = torch.get_default_dtype()

#         graph_data = {
#             "nodes": torch.tensor(atoms.get_atomic_numbers()),
#             "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
#             "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
#             "edges": torch.tensor(edges),
#             "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
#             "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
#             "num_edges": torch.tensor(edges.shape[0])
#         }

#         return graph_data

#     def get_edges_simple(self, atoms):
#         # Compute distance matrix
#         pos = atoms.get_positions()
#         dist_mat = scipy.spatial.distance_matrix(pos, pos)

#         # Build array with edges and edge features (distances)
#         valid_indices_bool = dist_mat < self.cutoff
#         np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
#         edges = np.argwhere(valid_indices_bool)  # num_edges x 2
#         edges_displacement = np.zeros((edges.shape[0], 3))

#         return edges, edges_displacement






class GraphMaker:
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, atoms, trial_poses=None):
        return atoms_and_probe_sample_to_graph_dict(atoms, trial_poses, self.cutoff)



def atoms_and_probe_sample_to_graph_dict(atoms, trial_poses, cutoff):
    default_type = torch.get_default_dtype()

    atom_edges, atom_edges_displacement, neighborlist, inv_cell_T = atoms_to_graph(atoms, cutoff)

    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
        "num_nodes": torch.tensor(atoms.get_number_of_atoms()),
        "num_atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0).shape[0]),
        "atom_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
        "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
    }

    if trial_poses is not None:
        probe_edges, probe_edges_displacement = probes_to_graph(
            atoms, trial_poses, cutoff, neighborlist=neighborlist, inv_cell_T=inv_cell_T
        )

        if not probe_edges:
            probe_edges = [np.zeros((0, 2), dtype=np.int_)]
            probe_edges_displacement = [np.zeros((0, 3), dtype=np.int_)]

        res.update({
            "probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0)),
            "probe_edges_displacement": torch.tensor(
                np.concatenate(probe_edges_displacement, axis=0), dtype=default_type
            ),
            "num_probe_edges": torch.tensor(np.concatenate(probe_edges, axis=0).shape[0]),
            "num_probes": torch.tensor(trial_poses.shape[0]),
            "probe_xyz": torch.tensor(trial_poses, dtype=default_type),
        })


    return res


def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if (
        np.any(atoms.get_cell().lengths() <= 0.0001)
        or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T



def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None):
    probe_edges = []
    probe_edges_displacement = []

    num_probes = probe_pos.shape[0]
    probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
    atoms_with_probes = atoms.copy()
    atoms_with_probes.extend(probe_atoms)
    atomic_numbers = atoms_with_probes.get_atomic_numbers()

    if (
        np.any(atoms.get_cell().lengths() <= 0.0001)
        or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

    results = [neighborlist.get_neighbors(i+len(atoms), cutoff) for i in range(num_probes)]

    atom_positions = atoms.get_positions()
    for i, (neigh_idx, neigh_vec, _) in enumerate(results):
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]
        self_index = np.ones_like(neigh_atoms) * i
        edges = np.stack((neigh_atoms, self_index), axis=1)

        neigh_pos = atom_positions[neigh_atoms]
        this_pos = probe_pos[i]

        neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        probe_edges.append(edges)
        probe_edges_displacement.append(neigh_origin_scaled)

    return probe_edges, probe_edges_displacement







