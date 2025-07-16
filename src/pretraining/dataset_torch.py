
import os, pickle, logging
from typing import List
from time import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.agents.base import AbstractActorCritic
import src.pretraining.action_decom as decom
from src.rl.spaces import ObservationSpace, ActionSpace
from src.tools import util
from src.tools.arg_parser_pretrain import build_default_argparser_pretrain
from src.tools.model_util import ModelIO, build_model

from ase import Atoms


# TODO: 
# Define hp dataclass instead
# Make simple AtomsObject dataclass also (in case more info is needed than available in Atoms)
# Make AutoregressiveDataset and inherit from this if necessary,
#   e.x. MolGymDataset, ExplorerDataset, MaskedExplorerDataset, etc.


class MolGymDataset(Dataset):
    """A dataset class for expert trajectory processing in MolGym"""

    def __init__(
            self, 
            df_train: pd.DataFrame,
            model: AbstractActorCritic,
            observation_space: ObservationSpace,
            action_space: ActionSpace,
            config: dict
    ):
        super().__init__()

        self.df_train = df_train
        self.model = model
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, index):
        id = self.df_train.index[index]
        mol = self.df_train.loc[id]


        mol_dataset = self.config['mol_dataset']
        decom_method = self.config['decom_method']
        cutoff = self.config['decom_cutoff']
        shuffle = self.config['decom_shuffle']
        mega_shuffle = self.config['decom_mega_shuffle']
        hydrogen_delay = self.config['hydrogen_delay']
        no_hydrogen_focus = self.config['no_hydrogen_focus']

        pos, elements, symbols, formula = mol['pos'], mol['atomic_nums'], mol['atomic_symbols'], mol['formulas']

        # Apply transformations
        pos = decom.gaussian_perturbation(pos, sigma=0.05)
        pos = decom.recenter(pos, elements, formula, mol_dataset)

        # Decompose the molecule
        sorted_indices = decom.decompose_pos(elements, pos, decom_method=decom_method, cutoff=cutoff, 
                                             shuffle=shuffle, mega_shuffle=mega_shuffle, 
                                             hydrogen_delay=hydrogen_delay)
        if sorted_indices is None:
            return {'status': 'failed decomposition', 'original_index': id}
        
        # Rotate the molecule to the z-axis
        pos = decom.rotate_to_axis(pos, atom_index=sorted_indices[1], axis='z')

        # Reorder the atoms
        pos, elements, symbols = pos[sorted_indices, :], np.array(elements)[sorted_indices], symbols[sorted_indices]

        #################### Extract states, actions and returns ####################

        # Get RL actions
        actions = decom.pos_seq_to_actions_explorer(pos, elements, self.observation_space.zs, 
                                                    no_hydro_focus=no_hydrogen_focus, 
                                                    num_trials=self.config["num_trials"])

        # Get RL obs
        obs = self.get_rl_obs(pos, symbols)
        if obs is None:
            return {'status': 'failed action reconstruction', 'original_index': id}
    
        assert len(obs) == actions.shape[0], f"len(obs): {len(obs)}, actions.shape[0]: {actions.shape[0]}"
        return {'status': 'success', 'expert_obs': obs, 'expert_actions': actions}


    def get_rl_obs(self, pos, symbols):
        """ Get the RL observations for the entire trajectory """
        total_atoms = len(symbols)

        all_obs = []
        for t in range(0, pos.shape[0]):
            atoms = Atoms(positions=pos[0:t, :], symbols=symbols[0:t])
            n_atoms_to_go = total_atoms - t
            all_obs.append(self.observation_space.build(atoms=atoms, n_atoms_to_go=n_atoms_to_go))

        return all_obs

import random

class BasicCollator:
    """ 
        Collates single molecule trajectories into a single batch.
        Essentially just list_of_dicts -> dict_of_lists.
    """

    def __init__(self):
        pass

    def __call__(self, graphs: List[dict]):

        # for testing, i dont won't the first element to correspond to an empty graph, 
        # so lets just shuffle the list to make sure the data is not sorted
        random.shuffle(graphs)


        use_advantage = "expert_adv" in graphs[0]
        use_returns = "expert_rets" in graphs[0]

        expert_obs = []
        expert_actions = []

        expert_rets = []
        expert_adv = []
        for graph in graphs:
            if graph["status"] == "success":
                # print(f"len(graph['expert_obs']): {len(graph['expert_obs'])}")
                expert_obs.extend(graph["expert_obs"])
                expert_actions.append(graph["expert_actions"])

                if use_returns:
                    expert_rets.append(graph["expert_rets"])
                if use_advantage:
                    expert_adv.append(graph["expert_adv"])
        
        expert_actions = np.concatenate(expert_actions, axis=0)

        if use_returns:
            expert_rets = np.concatenate(expert_rets, axis=0)
        if use_advantage:
            expert_adv = np.concatenate(expert_adv, axis=0)
            # advantage normalization trick
            expert_adv = (expert_adv - expert_adv.mean()) / (expert_adv.std() + 1e-8)

        collated = {
            "obs": expert_obs,
            "act": expert_actions,
        }

        if use_returns:
            collated["rets"] = expert_rets
        if use_advantage:
            collated["adv"] = expert_adv

        return collated


def get_pretrain_dataloader(
    df_train: pd.DataFrame, 
    model: AbstractActorCritic, 
    observation_space: ObservationSpace, 
    action_space: ActionSpace, 
    config: dict
) -> DataLoader:
    """ Build pretraining data loader """

    molgym_dataset = MolGymDataset(df_train=df_train,
                                   model=model,
                                   observation_space=observation_space,
                                   action_space=action_space,
                                   config=config)
    logging.info(f"length of molgym_dataset for pretraining: {len(molgym_dataset)}")

    dataloader_params = {
        "batch_size": config['dataloader_bs'],
        "shuffle": True,
        "num_workers": config['dataloader_num_workers'],
        "pin_memory": False
    }

    return DataLoader(molgym_dataset, **dataloader_params, collate_fn=collate_molecules)




from src.pretraining.data_utils import get_train_and_eval_data

def get_config_pretrain() -> dict:
    parser = build_default_argparser_pretrain()
    args = parser.parse_args()
    config = vars(args)
    return config

if __name__ == '__main__':
    
    config = get_config_pretrain()
    config["train_mode"] = 'pretrain'
    config["reward_coefs"] = {'rew_abs_E': 1.0 , 'rew_valid': 3.0}
    config["model"] = 'painn'
    config["use_GMM"] = True

    n_atoms_min = 17
    n_atoms_max = 23
    n_test = 0


    zs, canvas_size, eval_formulas, train_formulas, df_train, benchmark_energies, _ = get_train_and_eval_data(
        config, n_atoms_min=n_atoms_min, n_atoms_max=n_atoms_max, n_test=n_test)

    print(f'df_train: {df_train.head()}')
    print(f'eval_formulas: {eval_formulas}')

    # eval_formulas = sorted(list(set(df_test['bag_repr'].values.tolist())))
    # benchmark_energies, _ = get_benchmark_energies(eval_formulas, mol_dataset)

    # print(f'eval_formulas: {eval_formulas}')
    # print(f'benchmark_energies: {benchmark_energies}')


    action_space = ActionSpace(zs=zs)
    observation_space = ObservationSpace(canvas_size=canvas_size, zs=zs)
    device = util.init_device(config['device'])
    model = build_model(config, observation_space=observation_space, action_space=action_space, device=device)
    model.set_device(device)
    optimizer=util.get_optimizer(name=config['optimizer'],
                                 learning_rate=config['learning_rate'],
                                 parameters=model.parameters()),

    print(f'optimizer: {optimizer[0]}')

    molgym_dataset = MolGymDataset(df_train=df_train,
                                   model=model,
                                   observation_space=observation_space,
                                   action_space=action_space,
                                   config=config)

    molgym_dataset.__getitem__(100)

    exit()

    dataloader_params = {
        "batch_size": config['dataloader_bs'],
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": False
    }
    data_loader = DataLoader(molgym_dataset, **dataloader_params, collate_fn=collate_molecules)



    start_time = time()
    for data_batch in data_loader:
        print(len(data_batch['obs']))
        # loss_info = optimize_agent(ac=model, data_batch=data_batch, optimizer=optimizer[0], vf_coef=config['vf_coef'], 
        #                            entropy_coef=config['entropy_coef'], beta=config['beta_MARWIL'], device=device, 
        #                            gradient_clip=config['gradient_clip'])

        # print(loss_info)
        break

    print(f"Time taken: {time() - start_time}")

    exit()