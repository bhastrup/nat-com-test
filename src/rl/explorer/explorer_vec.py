from typing import List, Tuple
import copy
import dataclasses
import gym
import numpy as np

from src.rl.env_container import SimpleEnvContainer
from src.rl.spaces import ObservationType

from src.omdiff.data.components import AtomsData, Batch
from src.omdiff.data.components import collate_data
from src.omdiff.data.components.transforms.base import Transform
from src.omdiff.models.corrector.model import Corrector
from src.omdiff.models.corrector.sampling import BaseCorrectorSampler
from src.omdiff.data.utils import convert_batch_to_atoms, save_images

import torch


@dataclasses.dataclass
class DiffusionHParams:
    scale_positions: float = 1.0


class GNNEnvContainer(SimpleEnvContainer):
    """
    A container for environments that use a shared GNN model for the env.step() function,
    similar to what is done on the agent side.
    """
    def __init__(
            self, 
            environments: List[gym.Env], 
            batch_transforms: Transform,
            sampler: BaseCorrectorSampler,
            edm: Corrector,
    ):
        super().__init__(environments)
        self.environments = environments
        self.actions = None

        self.batch_transforms = batch_transforms
        self.sampler = sampler
        self.edm = edm

        self.node_labels = self.environments[0].action_space.zs
        self.keep_intermediate_states = False

        self.device = next(edm.parameters()).device

    def step_wait(self):
        assert self.actions and len(self.environments) == len(self.actions)

        # a) First implement the explorer decisions on the environments (to arrive at s*_t)
        for env, action in zip(self.environments, self.actions):
            env.apply_explorer(action, inject_noise=True)

        # b) Then apply the Corrector agent
        (batch, active_idx) = self.create_state_batch()
        if batch is not None:
            time_step_dict = self.denoise_with_corrector(batch)
            corrector_atoms = time_step_dict[0]
            # Update positions in current_atoms
            for idx, outer_idx in enumerate(active_idx):
                env = self.environments[outer_idx]
                atoms = corrector_atoms[idx]
                env.apply_corrector(updated_atoms=atoms)

        # c) Obtain the next state s_{t+1}, rewards, done flag and info
        results = [env.post_step() for env in self.environments]
        obs_list, rewards, done_list, infos = zip(*results)

        return obs_list, np.array(rewards), np.array(done_list), infos


    def create_state_batch(self) -> Tuple[Batch, List[int]]:
        # Create a batch of AtomsData objects
        active_idx = []
        atoms_data_list = []
        for (i, env) in enumerate(self.environments):
            # We don't need to predict on structures with n_atoms < 2.
            if len(env.current_atoms) < 2:
                continue
            
            active_idx.append(i)
            atoms = env.current_atoms
            num_atoms_to_go = env.current_num_atoms_to_go

            atoms_data = AtomsData(
                node_features=torch.tensor(atoms.get_atomic_numbers()).unsqueeze(1),
                node_positions=torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype()),
            )
            setattr(
                atoms_data, 
                "num_atoms_to_go",
                torch.tensor([num_atoms_to_go], dtype=torch.get_default_dtype())
            )
            atoms_data_list.append(atoms_data)

        if len(atoms_data_list) > 0:
            batch = self.batch_transforms(atoms_data_list)
            batch = collate_data(atoms_data_list).to(self.device)
        else:
            batch = None

        return (batch, active_idx)

    def denoise_with_corrector(self, batch):

        # Sample trajectories
        time_step_dict: dict[int, list] = {}
        for t, samples in self.sampler.generate(
                batch=batch, 
                edm=self.edm,
                yield_every=-1, # Just yield the last!
                yield_init=False,
                perturb_init=False,
        ):
            if t not in time_step_dict:
                time_step_dict[t] = []
            samples_cpu = copy.deepcopy(samples).to(
                "cpu"
            )  # copy needed as samples is used by the generator
            time_step_dict[t].extend(
                convert_batch_to_atoms(
                    samples_cpu,
                    node_labels=self.node_labels[1:],
                    scale_positions=1.0,
                )
            )

        return time_step_dict
