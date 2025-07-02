


from typing import Tuple, List, Optional

import ase
import ase.data
import numpy as np
import torch
import torch.distributions
from torch import nn
from torch.nn.functional import softmax

from src.agents.base import AbstractActorCritic
from src.agents.internal import zmat
from src.agents.modules import MLP, masked_softmax, to_one_hot
from src.tools.util import to_numpy, fibonacci_sphere


from src.agents.painn import layer_painn as layer
from src.agents.painn import data_painn

from src.rl.explorer.to_graph import GraphMaker
from src.rl.explorer.nets import (
    PainnProbeMessageModel, 
    PainnAtomRepresentationModel
)

from src.rl.explorer.spaces import (
    ObservationSpace, 
    ObservationType, 
    ActionType, 
    ActionSpace
)


def nan_to_num_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in {module}. Replacing NaNs with 0.")
        # Replace NaNs, positive infinity, and negative infinity values in the output
        output = torch.nan_to_num(output, nan=0.0, posinf=1e10, neginf=-1e10)
    return output


class ExplorerAC(AbstractActorCritic):
    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_max_distance: Tuple[float, float],
        network_width: int,
        num_interactions: int,
        cutoff: float,
        hydrogen_delay: bool,
        no_hydrogen_focus: bool,
        rms_norm_update: bool,
        num_trials: int,
        device: torch.device,
    ):
        # Internal action: stop, focus, element, proposal_site
        self.num_subactions = 3 # For make_atomic_tensors() (not counting "stop" token here)

        super().__init__(observation_space=observation_space, action_space=action_space)
        self.device = device
        self.hydrogen_delay = hydrogen_delay
        self.no_hydrogen_focus = no_hydrogen_focus

        self.num_atoms = self.observation_space.canvas_space.size
        self.num_zs = len(self.observation_space.zs)

        self.num_afeats = network_width // 2
        self.num_latent_beta = network_width // 4
        self.num_latent = self.num_afeats + self.num_latent_beta

        # PaiNN variables:

        #self.transformer = data_painn.TransformAtomsObjectsToGraphXyz(cutoff=cutoff)
        #self.old_transformer = data_painn.TransformAtomsObjectsToGraphXyzTester(cutoff=cutoff)
        #self.deep_dft_transformer = ReferenceMethod(cutoff=cutoff)
        #self.our_transformer = AtomsToMoleculeGraph(cutoff=cutoff)
        self.painn_transformer = GraphMaker(cutoff=cutoff)
        
        self.hidden_state_size = network_width // 2
        self.cutoff = cutoff
        self.distance_embedding_size = 20

        num_embeddings = 119  # atomic numbers + 1


        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            self.hidden_state_size,
            self.cutoff,
            self.distance_embedding_size,
        )

        # Setup probe model for choosing between next atom trial positions
        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            self.hidden_state_size,
            self.cutoff,
            self.distance_embedding_size,
        )


        # MolGym neural networks
        self.phi_num_atoms_to_go = MLP(
            input_dim=1,
            output_dims=(network_width, self.num_latent_beta),
        )

        self.phi_focus = MLP(
            input_dim=self.num_latent,
            output_dims=(network_width, 1),
        )

        self.phi_element = MLP(
            input_dim=self.num_latent,
            output_dims=(network_width, self.num_zs),
        )

        # New neural network for choosing between trial positions
        self.num_trials = num_trials
        self.phi_trial = MLP(
            input_dim=self.num_latent + self.num_zs,
            output_dims=(network_width, 1),
        )

        self.critic = MLP(
            input_dim=self.num_latent,
            output_dims=(network_width, network_width, 1),
        )

        self.to(device)

        # Hook to replace NaNs with 0
        over_write_nan = False
        if over_write_nan:
            for module in self.modules():
                module.register_forward_hook(nan_to_num_hook)


    def set_device(self, device: torch.device):
        self.device = device
        self.pin = (device == torch.device("cuda"))

    @staticmethod
    def _sample_atom_around_focus(atoms, focus):
        positions = [atom.position for atom in atoms]

        if len(atoms) > 0:
            # Sample position randomly around focus
            guassian_noise = np.random.randn(3)

            # Constrain guassian noise vector length to be larger than 0.2 and smaller than 1.5
            bounds = [0.3, 1.5]
            length = np.linalg.norm(guassian_noise)
            if length < bounds[0]:
                guassian_noise = guassian_noise / length * bounds[0]
            elif length > bounds[1]:
                guassian_noise = guassian_noise / length * bounds[1]

            position = positions[focus] + guassian_noise
        else:
            position = np.array([0., 0., 0.])
        
        return position

    def _place_atom_at_trial_position(self, positions, focus, trial_idx):
        """
        Places an atom at a trial position.
        """
        trial_poses = torch.tensor(fibonacci_sphere(self.num_trials), dtype=torch.float32) + positions[focus]        
        return trial_poses[trial_idx]
        

    def to_action_space(self, action: torch.Tensor, observation: ObservationType) -> ActionType:
        stop, focus, element, trial_idx = to_numpy(action)
    
        if stop:
            return self.action_space.from_atom(ase.Atoms())

        # Round to obtain discrete subactions
        focus = int(round(focus))
        element = int(round(element))
        trial_idx = int(round(trial_idx))

        atoms, num_atoms_to_go = self.observation_space.parse(observation)

        if len(atoms) > 0:
            positions = atoms.get_positions()
            position = self._place_atom_at_trial_position(positions, focus, trial_idx)
        else:
            position = np.zeros((3,))

        atomic_number_index = self.action_space.zs.index(self.observation_space.zs[element])
        return atomic_number_index, position

    def make_atomic_tensors(
        self, observations: List[ObservationType]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        features = torch.zeros(size=(len(observations), self.num_atoms, self.num_afeats),
                               dtype=torch.float32,
                               device=self.device)
        focus_mask = torch.zeros(size=(len(observations), self.num_atoms), dtype=torch.int, device=self.device)
        focus_mask_next = torch.zeros(size=(len(observations), self.num_atoms), dtype=torch.int, device=self.device)
        # element_count = torch.zeros(size=(len(observations), self.num_zs), dtype=torch.float32, device=self.device)
        action_mask = torch.zeros(size=(len(observations), self.num_subactions), dtype=torch.float32, device=self.device)
        num_atoms_to_go = torch.zeros(size=(len(observations), 1), dtype=torch.float32, device=self.device)

        element_mask = torch.ones(size=(len(observations), self.num_zs), dtype=torch.int, device=self.device)
        element_mask[:, 0] = 0

        n_atoms_list = []
        graph_states = []
        worker_index_all = []
        for i, obs in enumerate(observations):
            # Get Atoms() object from observation
            atoms, num_atoms_to_go_value = self.observation_space.parse(obs)
            n_atoms_list.append(len(atoms))

            if len(atoms) > 0:
                # Transform to graph dictionary
                graph_state = self.painn_transformer(atoms, trial_poses=None)
                graph_states.append(graph_state)
                worker_index_all.append(i)

                focus_mask[i, :len(atoms)] = 1
                focus_mask_next[i, :len(atoms) + 1] = 1
            else:
                focus_mask[i, :1] = 1  # focus null-atom
                focus_mask_next[i, :2] = 1

            num_atoms_to_go[i] = num_atoms_to_go_value

            # Mask out subactions:
            action_mask[i, 0] = len(atoms) >= 1         # focus
            action_mask[i, 1] = 1.0                     # element
            action_mask[i, 2] = len(atoms) >= 1         # 3D placement

        # Get PaiNN embeddings for all observations
        mpnn_output = {}
        if len(graph_states) > 0:
            batch_host = data_painn.collate_atomsdata(graph_states, pin_memory=self.pin)
            batch = {
                k: v.to(device=self.device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            # nodes_scalar, _, edge_offset = self._get_painn_embeddings(batch)
            nodes_list_scalar, nodes_list_vector, edge_offset = self.atom_model(batch)
            mpnn_output['atom_representation_scalar'] = nodes_list_scalar # watch out. Here we stay in the batch catenated tensor.
            mpnn_output['atom_representation_vector'] = nodes_list_vector

        for i, obs in enumerate(observations):
            n_atoms = n_atoms_list[i]
            if n_atoms > 0:
                worker_index = worker_index_all.index(i)
                slc = slice(edge_offset[worker_index], edge_offset[worker_index]+batch['num_nodes'][worker_index])
                features[i, :n_atoms, :] = nodes_list_scalar[-1][slc, :]

        # # Or we could have done this:
        # nodes_scalar_copy = nodes_scalar.clone()
        # features_copy = features.clone() # Should be moved above 

        # features_pad_and_stack = data_painn.pad_and_stack(
        #     torch.split(
        #         nodes_scalar_copy,
        #         list(batch['num_nodes'].detach()),
        #         dim=0,
        #     )
        # )
        # out_shape = features_pad_and_stack.shape
        # features_copy[torch.tensor(worker_index_all), :out_shape[1], :] = features_pad_and_stack
        # assert torch.allclose(features, features_copy), f"shape mismatch: {features.shape} != {features_copy.shape}"
        # print(f'Successfully used pad_and_stack to get features. Exiting.')


        return (
            features,           # n_obs x n_atoms x n_afeats
            focus_mask,         # n_obs x n_atoms
            focus_mask_next,    # n_obs x n_atoms
            element_mask,       # n_obs x num_elements
            action_mask,        # n_obs x num_subactions
            num_atoms_to_go,    # n_obs x 1
            mpnn_output         # dict of tensors. keys: 'atom_representation_scalar', 'atom_representation_vector'
        )


    def probe_features(self, observations: List[ObservationType], focus: torch.Tensor, 
                       element: torch.Tensor, mpnn_output: dict) -> torch.Tensor:
        features = torch.zeros(size=(len(observations), self.num_trials, self.num_afeats), dtype=torch.float32, device=self.device)
        focus, element = to_numpy(focus), to_numpy(element)

        n_atoms_list = []
        graph_states = [] # list of graph states incl. probe atoms
        worker_index_all = []
        for i, obs in enumerate(observations):
            atoms, num_atoms_to_go_value = self.observation_space.parse(obs)
            n_atoms_list.append(len(atoms))

            # Get probe graph states
            if len(atoms) > 0: 
                # OBS: We make sure to use the same condition (len(atoms) > 0) as in make_atomic_tensors(), despite not caring about n_atoms==1, 
                # as all directions are equally good. Then we can reuse the exact same nodes_scalar and nodes_vector tensors.
                worker_index_all.append(i)
                positions = [atom.position for atom in atoms]
                focus_atom=int(round(focus[i, 0]))
                new_element = int(round(element[i, 0])) # not an actual element, but a number from {0, 1, 2, ... len(self.observation_space.zs) - 1}
                # new_element_real = self.observation_space.zs[new_element]
                # new_symbol_real = ase.data.chemical_symbols[new_element_real]
                trial_poses = fibonacci_sphere(self.num_trials) + positions[focus_atom]
                graph_dict_with_probes = self.painn_transformer(atoms, trial_poses=trial_poses)
                graph_states.append(graph_dict_with_probes)

        # Get PaiNN embeddings for all observations
        if len(graph_states) > 0:
            batch_host = data_painn.collate_atomsdata(graph_states, pin_memory=self.pin)

            batch = {
                k: v.to(device=self.device, non_blocking=True)
                for (k, v) in batch_host.items()
            }
            probe_state_scalar, probe_state_vector, edge_probe_offset = self.probe_model(
                input_dict=batch,
                atom_representation_scalar=mpnn_output['atom_representation_scalar'],
                atom_representation_vector=mpnn_output['atom_representation_vector'],
                new_elements=torch.tensor(element[worker_index_all], device=self.device).squeeze(-1)
            )
            edge_probe_offset = edge_probe_offset.squeeze(1)[:, 1]
            for i, obs in enumerate(observations):
                n_atoms = n_atoms_list[i]
                if n_atoms > 0:
                    worker_index = worker_index_all.index(i)
                    slc = slice(edge_probe_offset[worker_index], edge_probe_offset[worker_index]+batch['num_probes'][worker_index])
                    features[i, :, :] = probe_state_scalar[slc, :]

        return features
    

    def step(self, observations: List[ObservationType], actions: Optional[np.ndarray] = None) -> dict:
        # atomic_feats: n_obs x n_atoms x n_afeats
        # focus_mask: n_obs x n_atoms
        # focus_mask: n_obs x n_atoms
        # element_count: n_obs x n_zs

        (
            atomic_feats, 
            focus_mask_perceive, 
            focus_mask_next, 
            element_mask, 
            action_mask, 
            num_atoms_to_go,
            mpnn_output
        ) = self.make_atomic_tensors(observations)

        # stop: this agent does not stop
        stop = torch.zeros(size=(len(observations), 1), dtype=torch.float, device=self.device)

        # latent states bag
        # latent_bag = self.phi_beta(num_atoms_to_go)
        latent_bag = self.phi_num_atoms_to_go(num_atoms_to_go) # n_obs x n_latent_beta

        # latent representation of atoms and bag
        latent_bag_tiled = latent_bag.unsqueeze(1)  # n_obs x 1 x n_latent_beta
        latent_bag_tiled = latent_bag_tiled.expand(-1, self.num_atoms, -1)  # n_obs x n_atoms x n_latent_beta

        latent_states = torch.cat([atomic_feats, latent_bag_tiled], dim=-1)  # n_obs x n_atoms x (n_afeats + n_latent_beta)

        # Focus
        focus_logits = self.phi_focus(latent_states)  # n_obs x n_atoms x 1
        focus_logits = focus_logits.squeeze(-1)  # n_obs x n_atoms

        focus_p = masked_softmax(focus_logits, mask=focus_mask_perceive.bool())  # n_obs x n_atoms        
        focus_p = torch.nan_to_num(focus_p, nan=0.0, posinf=1e10, neginf=-1e10)
        focus_dist = torch.distributions.Categorical(probs=focus_p)

        # Cast action to Tensor
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device)

        # focus: n_obs x 1
        if actions is not None:
            focus = torch.round(actions[:, 1:2]).long()
        elif self.training:
            focus = focus_dist.sample().unsqueeze(-1)
        else:
            focus = torch.argmax(focus_p, dim=-1).unsqueeze(-1)

        focus_oh = to_one_hot(focus, num_classes=self.num_atoms, device=self.device)  # n_obs x n_atoms

        # Focused atom is a hard (one-hot) selection over atoms
        focused_atom = (latent_states.transpose(1, 2) @ focus_oh[:, :, None]).squeeze(-1)  # n_obs x n_latent

        # Element
        element_logits = self.phi_element(focused_atom)  # n_obs x n_zs
        element_p = masked_softmax(element_logits, mask=element_mask.bool())  # n_obs x n_zs
        element_p = torch.nan_to_num(element_p, nan=0.0, posinf=1e10, neginf=-1e10)
        element_dist = torch.distributions.Categorical(probs=element_p)

        # element: n_obs x 1
        if actions is not None:
            element = torch.round(actions[:, 2:3]).long()
        elif self.training:
            element = element_dist.sample().unsqueeze(-1)
        else:
            element = torch.argmax(element_p, dim=-1).unsqueeze(-1)

        element_oh = to_one_hot(element, self.num_zs, device=self.device)  # n_obs x n_zs
        element_oh = element_oh.unsqueeze(1)  # n_obs x 1 x n_zs
        element_oh = element_oh.expand(-1, self.num_trials, -1)  # n_obs x n_trials x n_zs

        # Trial
        trial_latents = self.probe_features(observations, focus, element, mpnn_output) # n_obs x n_trials x n_afeats
        latent_bag_tiled = latent_bag.unsqueeze(1)  # n_obs x 1 x n_latent_beta
        latent_bag_tiled = latent_bag_tiled.expand(-1, self.num_trials, -1)  # n_obs x n_trials x n_latent_beta
        trial_latents = torch.cat([trial_latents, latent_bag_tiled, element_oh], dim=-1)  # n_obs x n_trials x (n_afeats + n_latent_beta + n_zs)
        trial_logits = self.phi_trial(trial_latents)  # n_obs x n_trials
        trial_probs = softmax(trial_logits.squeeze(2), dim=-1)  # n_obs x n_trials
        trial_probs = torch.nan_to_num(trial_probs, nan=0.0, posinf=1e10, neginf=-1e10)
        trial_dist = torch.distributions.Categorical(probs=trial_probs)

        # trial: n_obs x 1
        if actions is not None:
            trial = torch.round(actions[:, 3:4]).long()
        elif self.training:
            trial = trial_dist.sample().unsqueeze(-1)
        else:
            trial = torch.argmax(trial_probs, dim=-1).unsqueeze(-1)

        if actions is None:
            actions = torch.cat([stop, focus.float(), element.float(), trial.float()], dim=-1)

        # Critic
        weights = focus_mask_perceive.unsqueeze(-1).float()  # n_obs x n_atoms x 1
        weights = weights.transpose(1, 2)  # n_obs x 1 x n_atoms
        sum_atomic_feats = (weights @ atomic_feats).squeeze(1)  # n_obs x n_afeats
        # mean_atomic_feats = sum_atomic_feats / torch.sum(focus_mask, dim=-1, keepdim=True)
        v = self.critic(torch.cat([sum_atomic_feats, latent_bag], dim=-1))

        # Log probabilities
        log_prob_list = [
            focus_dist.log_prob(focus.squeeze(-1)).unsqueeze(-1),
            element_dist.log_prob(element.squeeze(-1)).unsqueeze(-1),
            trial_dist.log_prob(trial.squeeze(-1)).unsqueeze(-1),
        ]
        log_prob = torch.cat(log_prob_list, dim=-1)

        # Mask
        log_prob = log_prob * action_mask

        # Entropies
        entropy_list = [
            focus_dist.entropy().unsqueeze(-1),
            element_dist.entropy().unsqueeze(-1),
            trial_dist.entropy().unsqueeze(-1),
        ]
        entropy = torch.cat(entropy_list, dim=-1)

        # Mask
        entropy = entropy * action_mask


        return {
            'a': actions,  # n_obs x n_subactions
            'logp': log_prob.sum(dim=-1, keepdim=False),  # n_obs
            'ent': entropy.sum(dim=-1, keepdim=False),  # n_obs
            'v': v.squeeze(-1),  # n_obs

            # Actions in action space
            'actions': [self.to_action_space(a, o) for a, o in zip(actions, observations)],
        }
