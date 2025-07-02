from typing import List, Dict
import math
import ase
import torch
from torch import nn
import src.rl.explorer.probe_layer as layer


# Based on https://github.com/peterbjorgensen/DeepDFT/blob/main/densitymodel.py

class PainnDensityModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size=30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

    def forward(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation_scalar, atom_representation_vector)
        return probe_result



class PainnAtomRepresentationModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.PaiNNInteraction(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def forward(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        nodes_list_scalar = []
        nodes_list_vector = []
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector, edge_offset
## Why does it return all the tensors?


class PainnProbeMessageModel(nn.Module):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        init_scalars_as_new_element: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        self.init_scalars_as_new_element = init_scalars_as_new_element
        if self.init_scalars_as_new_element: 
            # Atom embeddings
            self.atom_embeddings = nn.Embedding(
                len(ase.data.atomic_numbers), self.hidden_state_size
            )
            # TODO: Then we could go back AC and remove new_elements from phi_trial nn input

        # Setup interaction networks
        self.message_layers = nn.ModuleList(
            [
                layer.PaiNNInteractionOneWay(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_function = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, 1),
        )

    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        atom_representation_scalar: List[torch.Tensor],
        atom_representation_vector: List[torch.Tensor],
        new_elements: torch.Tensor,
    ):
        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(
            input_dict["probe_xyz"], input_dict["num_probes"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_probes"].device),
                    input_dict["num_probes"][:-1],
                )
            ),
            dim=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = torch.cat((edge_offset, edge_probe_offset), dim=2)
        probe_edges = input_dict["probe_edges"] + edge_probe_offset
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # Compute edge distances
        probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        if self.init_scalars_as_new_element:
            # Prepare initial scalar features of probes sites using next atom element.
            max_num_probes = input_dict["num_probes"].max()
            new_elements_tiled = new_elements[:, None, None].expand(-1, max_num_probes, -1) # n_obs x max_num_probes x 1
            new_elements_scalar = layer.unpad_and_cat(new_elements_tiled, input_dict["num_probes"]).squeeze(-1) # n_total_probes
            probe_state_scalar = self.atom_embeddings(new_elements_scalar) # n_total_probes x hidden_state_size
        else:
            probe_state_scalar = torch.zeros(
                (torch.sum(input_dict["num_probes"]), self.hidden_state_size),
                device=atom_representation_scalar[0].device,
            )

        probe_state_vector = torch.zeros(
            (torch.sum(input_dict["num_probes"]), 3, self.hidden_state_size),
            device=atom_representation_scalar[0].device,
        )

        # Apply interaction layers
        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):
            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        return probe_state_scalar, probe_state_vector, edge_probe_offset

        # Restack probe states
        probe_output = self.readout_function(probe_state_scalar).squeeze(1)
        probe_output = layer.pad_and_stack(
            torch.split(
                probe_output,
                list(input_dict["num_probes"].detach().cpu().numpy()),
                dim=0,
            )
            # torch.split(probe_output, input_dict["num_probes"], dim=0)
            # probe_output.reshape((-1, input_dict["num_probes"][0]))
        )

        return probe_output