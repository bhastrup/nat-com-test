from pathlib import Path
from typing import List, Dict, Set, Optional
import os
import logging
import random

from ase import Atoms
import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dscribe.descriptors import SOAP

from src.performance.energetics import EnergyUnit
from src.agents.painn.agent import PainnAC
from src.tools.model_util import ModelIO
from src.data.reference_dataloader import ReferenceDataLoader
from src.tools import util




class SOAPProjector:
    """ Projects data into 2D space using PCA or similar methods.
        In particular we want to embed smiles strings into 2D space."""
    def __init__(self, mol_dataset: str='qm7', data_dir: Path=None, formulas: List[str]=None):
        self.mol_dataset = mol_dataset
        self.ref_data = ReferenceDataLoader(data_dir=data_dir).load_and_polish(
            mol_dataset, EnergyUnit.EV, fetch_df=True)
        
        if formulas:
            # Keep only the formulas in the list
            self.ref_data.df = self.ref_data.df[self.ref_data.df['bag_repr'].isin(formulas)] # change!
            self.ref_data.df.reset_index(drop=True, inplace=True)
            self.ref_data.smiles = {formula: smiles for formula, smiles in self.ref_data.smiles.items() if formula in formulas}

        # Get the embedder
        species = ["H", "C", "O", "N", "S"]
        rcut = 6.0
        nmax = 8
        lmax = 6

        self.embedder = SOAP(
            species=species,
            periodic=False,
            r_cut=rcut,
            n_max=nmax,
            l_max=lmax,
            average="outer",
            sparse=False
        )

        seed=42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.smiles_to_2d = self._fit_pca()

    def _embed(self, atoms_objects: List[Atoms]) -> np.ndarray:
        return self.embedder.create(
            system=atoms_objects,
            centers=None,
            n_jobs=1,
            verbose=False
        )

    def _get_ref_smiles(self) -> List[str]:
        ref_smiles = self.ref_data.smiles.copy()
        return [smiles for smiles_list in ref_smiles.values() for smiles in smiles_list]

    def _dataframe_to_atoms(self, df: pd.DataFrame) -> np.ndarray:
        # Obtain cols "pos" and "atomic_symbols"
        pos = df['pos'].values
        atomic_symbols = df['atomic_symbols'].values

        # Get observations
        atoms_objects = []
        for i in range(len(pos)):
            atoms = Atoms(positions=pos[i], symbols=atomic_symbols[i])
            atoms_objects.append(atoms)
        
        return atoms_objects

    def _fit_pca(self) -> Dict[str, np.ndarray]:
        """ Fit PCA / TSNE projector on the latent states (SOAP) of the reference data. """

        df = self.ref_data.df.copy()
        ref_obs = self._dataframe_to_atoms(df)
        soap_vectors = self._embed(ref_obs)
        self.pca = TSNE(n_components=2, perplexity=5) # PCA(n_components=2)

        print(f"Fitting PCA / TSNE dim reducer on SOAP features of shape: {soap_vectors.shape}")
        projs = self.pca.fit_transform(soap_vectors)
        print(f"PCA / TSNE fitted.")
        return {smiles: proj for smiles, proj in zip(df['SMILES'], projs)}

    def _latent_to_2d(self, latent_states: np.ndarray) -> np.ndarray:
        """ Project the latent states into 2D space."""
        logging.warning("Deprecated method. Use smiles_to_2d instead.")
        
        assert hasattr(self, 'pca'), "PCA object not found. Fit PCA first."
        return self.pca.transform(latent_states)

    def __call__(self, smiles_set: Set[str], find_complement_set: bool=False, use_fitted_projector: bool=True) -> np.ndarray:
        # Keep only intersection with ref_smiles
        ref_smiles = self._get_ref_smiles()
        smiles_set = {smiles for smiles in smiles_set if smiles in ref_smiles}

        if use_fitted_projector:
            assert hasattr(self, 'smiles_to_2d'), "Fitted projector not found. Fit PCA first."
            if not find_complement_set: 
                return np.array([self.smiles_to_2d[smiles] for smiles in smiles_set])
            else:
                return np.array([self.smiles_to_2d[smiles] for smiles in ref_smiles if smiles not in smiles_set])

        # Find rows in df where the smiles is in the smiles_set
        df = self.ref_data.df.copy()

        if not find_complement_set:
            mask = df['SMILES'].isin(smiles_set)
        else:
            mask = ~df['SMILES'].isin(smiles_set)

        df = df[mask]
        assert len(df['SMILES']) == len(set(df['SMILES'])), \
            "Reference data contains duplicate SMILES. Cannot obtain 3d coordinates."
        
        observations = self._dataframe_to_atoms(df)
        if not observations:
            return np.array([])

        soap_states = self._embed(observations)
        print(f"soap_states.shape: {soap_states.shape}")
        
        return self._latent_to_2d(soap_states)



class PCAProjector:
    """ Projects data into 2D space using PCA or similar methods.
        In particular we want to embed smiles strings into 2D space."""
    def __init__(self, mol_dataset: str='qm7', model_dir: Path=None, data_dir: Path=None):
        self.mol_dataset = mol_dataset

        self.ref_data = ReferenceDataLoader(data_dir=data_dir).load_and_polish(
            mol_dataset, EnergyUnit.EV, fetch_df=True)

        # Get the embedder
        model = get_model(model_dir=model_dir, newest=True)
        self.embedder = get_embedder(model)

        self.concat_bag = False
        dim_latent = self.embedder.num_latent if self.concat_bag else self.embedder.num_afeats

        seed=42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.smiles_to_2d = self._fit_pca()
        print("PCA projector fitted.")

    def _embed(self, obs) -> torch.Tensor:
        return self.embedder.get_embedding(obs, concat_bag = self.concat_bag)

    def _get_ref_smiles(self) -> List[str]:
        ref_smiles = self.ref_data.smiles.copy()
        return [smiles for smiles_list in ref_smiles.values() for smiles in smiles_list]

    def _dataframe_to_observations(self, df: pd.DataFrame) -> np.ndarray:
        # Obtain cols "pos" and "atomic_symbols"
        pos = df['pos'].values
        atomic_symbols = df['atomic_symbols'].values

        # Get observations
        observations = []
        for i in range(len(pos)):
            atoms = Atoms(positions=pos[i], symbols=atomic_symbols[i])
            obs = self.embedder.observation_space.build(
                atoms=atoms, 
                formula=((z, 0) for z in self.embedder.observation_space.bag_space.zs)
            )
            observations.append(obs)
        
        return observations

    def _fit_pca(self) -> Dict[str, np.ndarray]:
        """ Fit PCA projection on the latent states of the reference data. """

        df = self.ref_data.df.copy()
        ref_obs = self._dataframe_to_observations(df)
        ref_latent_states = self._embed(ref_obs).cpu().numpy()

        self.pca = PCA(n_components=2)
        projs = self.pca.fit_transform(ref_latent_states)
        return {smiles: proj for smiles, proj in zip(df['SMILES'], projs)}

    def _latent_to_2d(self, latent_states: torch.Tensor) -> np.ndarray:
        """ Project the latent states into 2D space."""
        logging.warning("Deprecated method. Use smiles_to_2d instead.")
        
        assert hasattr(self, 'pca'), "PCA object not found. Fit PCA first."
        latent_states = latent_states.cpu().numpy()
        return self.pca.transform(latent_states)

    def __call__(self, smiles_set: Set[str], find_complement_set: bool=False, use_fitted_projector: bool=True) -> np.ndarray:
        # Keep only intersection with ref_smiles
        ref_smiles = self._get_ref_smiles()
        smiles_set = {smiles for smiles in smiles_set if smiles in ref_smiles}

        if use_fitted_projector:
            assert hasattr(self, 'smiles_to_2d'), "Fitted projector not found. Fit PCA first."
            if not find_complement_set: 
                return np.array([self.smiles_to_2d[smiles] for smiles in smiles_set])
            else:
                return np.array([self.smiles_to_2d[smiles] for smiles in ref_smiles if smiles not in smiles_set])

        # Find rows in df where the smiles is in the smiles_set
        df = self.ref_data.df.copy()

        if not find_complement_set:
            mask = df['SMILES'].isin(smiles_set)
        else:
            mask = ~df['SMILES'].isin(smiles_set)

        df = df[mask]
        assert len(df['SMILES']) == len(set(df['SMILES'])), \
            "Reference data contains duplicate SMILES. Cannot obtain 3d coordinates."
        
        observations = self._dataframe_to_observations(df)
        if not observations:
            return np.array([])

        latent_states = self._embed(observations)
        return self._latent_to_2d(latent_states).cpu().numpy()





class PainnEmbedder(PainnAC):
    def get_embedding(self, observations, concat_bag = False):
        self.training = False
        with torch.no_grad():
            # Get the atomic features and sum them together
            atomic_feats, focus_mask, focus_mask_next, element_count, action_mask = self.make_atomic_tensors(observations)
            weights = focus_mask.unsqueeze(-1).float()  # n_obs x n_atoms x 1
            weights = weights.transpose(1, 2)  # n_obs x 1 x n_atoms
            sum_atomic_feats = (weights @ atomic_feats).squeeze(1)  # n_obs x n_afeats
            # mean_atomic_feats = sum_atomic_feats / torch.sum(focus_mask, dim=-1, keepdim=True)

            if concat_bag:
                latent_bag = self.phi_beta(element_count)
                latent_states = torch.cat([sum_atomic_feats, latent_bag], dim=-1)
            else:
                latent_states = sum_atomic_feats

        return latent_states


def get_embedder(model: PainnAC):

    # First copy action_space and observation_space and so on from model to embedder
    action_space = model.action_space
    observation_space = model.observation_space
    min_max_distance = (model.min_distance, model.max_distance)
    network_width = model.num_afeats * 2
    num_interactions = len(model.scalar_vector_update)
    cutoff = model.cutoff
    hydrogen_delay = model.hydrogen_delay
    no_hydrogen_focus = model.no_hydrogen_focus
    device = model.device
    rms_norm_update = model.scalar_vector_update[0].rms_norm

    embedder = PainnEmbedder(
        observation_space=observation_space,
        action_space=action_space,
        min_max_distance=min_max_distance,
        network_width=network_width,
        num_interactions=num_interactions,
        cutoff=cutoff,
        hydrogen_delay=hydrogen_delay,
        no_hydrogen_focus=no_hydrogen_focus,
        device=device,
        rms_norm_update=rms_norm_update
    )

    # Get the state dict from 'model' and load it into 'embedder'
    embedder.load_state_dict(model.state_dict())
    embedder.pin = model.pin

    return embedder

def get_model(model_dir: str, iter: Optional[int]=None, newest: bool=False) -> torch.nn.Module:
    model_handler = ModelIO(directory=model_dir, tag='Post Analysis')

    assert iter or newest and not (iter and newest), \
        "Either provide an iteration number or set newest to True, not both."

    if newest:
        import glob
        list_of_files = glob.glob(f"{model_dir}/*.model")
        iters = [int(file.split('steps-')[1].split('.model')[0]) for file in list_of_files]
        # sort list_of_files by iter
        list_of_files = [file for _, file in sorted(zip(iters, list_of_files))]
        selected_model_path = list_of_files[-1]
    else:
        names_containing_iter = [name for name in os.listdir(model_dir) if f'steps-{iter}' in name]
        selected_model_path = os.path.join(model_dir, names_containing_iter[0])


    device = util.init_device('cuda')
    # assert file exists
    assert os.path.exists(selected_model_path), f"Model file {selected_model_path} does not exist."
    model, start_num_steps = model_handler.load(device=device, path=selected_model_path)

    return model