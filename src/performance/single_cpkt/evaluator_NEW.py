
import os, json, logging
from typing import Dict, Tuple, List, Any, Union

import numpy as np
import pandas as pd

from ase import Atoms
from ase.io import read, write

from src.agents.base import AbstractActorCritic
from src.rl.rollouts import rollout_argmax_and_stoch
from src.performance.metrics import atom_list_to_df
from src.performance.single_cpkt.stats import (
    single_formula_metrics, get_global_metrics
)


def get_num_episodes(
    smiles: Dict[str, List[str]],
    num_episodes_const: int = None, 
    prop_factor: int = None, 
) -> Union[int, List[int]]:
    
    assert (num_episodes_const is None) != (prop_factor is None), \
        "One and only one of num_episodes_const and prop_factor must be provided."

    if num_episodes_const:
        num_episodes = num_episodes_const
    elif prop_factor:
        num_episodes = (np.array([len(v) for _, v in smiles.items()]) * prop_factor).tolist()
    
    episode_dict = {f: n for f, n in zip(smiles.keys(), num_episodes)}
    logging.info(f"Number of episodes: {episode_dict}")

    return num_episodes




class EvaluatorIO:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def save_smiles_to_json(self, smiles: Dict[str, List[str]]):
        for formula, smiles_list in smiles.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            with open(os.path.join(formula_dir, 'smiles.json'), 'w') as f:
                json.dump(smiles_list, f)

    def save_atoms_as_traj_file(self, final_atoms: Dict[str, List[Atoms]]):
        print(f"Saving atoms as traj files to {self.base_dir}")
        for formula, atom_list in final_atoms.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            write(os.path.join(formula_dir, 'atoms.traj'), atom_list)

    def read_atoms_as_traj_file(self, formulas: List[str]) -> Dict[str, List[Atoms]]:
        final_atoms = {}
        for formula in formulas:
            formula_path = os.path.join(self.base_dir, formula, 'atoms.traj')
            if not os.path.exists(formula_path):
                raise FileNotFoundError(f"No trajectory file found at {formula_path} for formula {formula}")
            atom_list = read(formula_path)
            final_atoms[formula] = [atom_list] if not isinstance(atom_list, list) else atom_list
        return final_atoms

    def save_dfs_to_csv(self, dfs: Dict[str, pd.DataFrame]):
        for formula, df in dfs.items():
            formula_dir = os.path.join(self.base_dir, formula)
            os.makedirs(formula_dir, exist_ok=True)
            df.to_csv(os.path.join(formula_dir, 'df.csv'), index=False)

    def load_dfs_from_csv(self, formulas: List[str]) -> Dict[str, pd.DataFrame]:
        formula_dfs = {}
        for formula in formulas:
            formula_dir = os.path.join(self.base_dir, formula)
            df_path = os.path.join(formula_dir, 'df.csv')
            if os.path.exists(df_path):
                formula_dfs[formula] = pd.read_csv(df_path)
            else:
                raise FileNotFoundError(f"No feature DataFrame found at {df_path} for formula {formula}")
        return formula_dfs

    def load_stats_and_smiles(self, formulas: List[str]) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
        formula_dfs = {}
        for formula in formulas:
            formula_dir = os.path.join(self.base_dir, formula)
            df_path = os.path.join(formula_dir, 'df.csv')
            smiles_path = os.path.join(formula_dir, 'smiles.json')

            if not os.path.exists(df_path):
                raise FileNotFoundError(f"No feature DataFrame found at {df_path} for formula {formula}")
            if not os.path.exists(smiles_path):
                raise FileNotFoundError(f"No smiles found at {smiles_path} for formula {formula}")

            df = pd.read_csv(os.path.join(formula_dir, 'df.csv'))
            smiles = json.load(open(os.path.join(formula_dir, 'smiles.json'), 'r'))
            formula_dfs[formula] = (df, smiles)
        return formula_dfs

    def save_global_metrics(self, global_metrics: Dict[str, Any]):
        with open(os.path.join(self.base_dir, 'global_metrics.json'), 'w') as f:
            json.dump(global_metrics, f)


class IOHandler:
    def __init__(self, io_handler: EvaluatorIO = None):
        self.io = io_handler
    
    def reset(self, new_base_dir: str, reference_smiles: Dict[str, list]):
        if self.io:
            self.io.base_dir = new_base_dir
            self.io.save_smiles_to_json(reference_smiles)

    def save_rollouts(self, rollouts):
        if self.io:
            self.io.save_atoms_as_traj_file(rollouts)

    def load_rollouts(self, eval_formulas):
        return self.io.read_atoms_as_traj_file(eval_formulas) if self.io else None

    def save_dfs(self, dfs):
        if self.io:
            self.io.save_dfs_to_csv(dfs)

    def load_dfs(self, eval_formulas):
        return self.io.load_dfs_from_csv(eval_formulas) if self.io else None

    def save_global_metrics(self, global_metrics):
        if self.io:
            self.io.save_global_metrics(global_metrics)



class EnvironmentManager:
    def __init__(self, eval_envs, reference_smiles: Dict[str, list], num_episodes_const: int, prop_factor: int):
        self.eval_envs = eval_envs
        self.reference_smiles = reference_smiles
        self.num_episodes = self.set_num_episodes(reference_smiles, num_episodes_const, prop_factor)
        self.eval_formulas = list(reference_smiles.keys())

    def set_num_episodes(self, reference_smiles: Dict[str, list], num_episodes_const: int, prop_factor: int):
        return get_num_episodes(reference_smiles, num_episodes_const, prop_factor)


class RolloutManager:
    def __init__(self, environment_manager: EnvironmentManager):
        self.environment_manager = environment_manager

    def perform_rollout(self, ac: AbstractActorCritic, rollouts_from_file: bool = False, io_handler: IOHandler = None):
        if rollouts_from_file:
            return io_handler.load_rollouts(self.environment_manager.eval_formulas) if io_handler else None
        else:
            rollouts = rollout_argmax_and_stoch(ac, self.environment_manager.eval_envs, self.environment_manager.num_episodes)
            io_handler.save_rollouts(rollouts) if io_handler else None
            return rollouts


class FeatureCalculator:
    def __init__(self, benchmark_energies_eval):
        self.benchmark_energies_eval = benchmark_energies_eval

    def calculate_features(self, rollouts, io_handler: IOHandler = None, features_from_file: bool = False):
        if features_from_file:
            return io_handler.load_dfs(self.benchmark_energies_eval.keys()) if io_handler else None
        else:
            dfs = {
                formula: atom_list_to_df(atom_list, benchmark_energies=self.benchmark_energies_eval)[0]
                for formula, atom_list in rollouts.items()
            }
            io_handler.save_dfs(dfs) if io_handler else None
            return dfs


class MetricsCalculator:
    def __init__(self, reference_smiles: Dict[str, list]):
        self.reference_smiles = reference_smiles

    def calculate_formula_metrics(self, dfs, io_handler: IOHandler = None):
        if io_handler:
            stats_and_smiles = io_handler.load_stats_and_smiles(dfs.keys())
        else:
            stats_and_smiles = {formula: (df, self.reference_smiles[formula]) for formula, df in dfs.items()}

        return {formula: single_formula_metrics(df, SMILES_db=smiles_list)
                for formula, (df, smiles_list) in stats_and_smiles.items()}

    def calculate_global_metrics(self, formula_metrics):
        return get_global_metrics(formula_metrics)


class SingleCheckpointEvaluator:
    def __init__(
        self, 
        eval_envs, 
        reference_smiles: Dict[str, list],
        benchmark_energies: Dict[str, float],
        io_handler: EvaluatorIO = None, 
        wandb_run: Any = None,
        num_episodes_const: int = 10,
        prop_factor: int = 100,
    ):
        self.environment_manager = EnvironmentManager(eval_envs, reference_smiles, num_episodes_const, prop_factor)
        self.io_handler = IOHandler(io_handler)
        self.rollout_manager = RolloutManager(self.environment_manager)
        self.feature_calculator = FeatureCalculator(benchmark_energies)
        self.metrics_calculator = MetricsCalculator(reference_smiles)
        self.wandb_run = wandb_run
        self.data = {}

    def reset(self, new_base_dir: str):
        self.io_handler.reset(new_base_dir, self.environment_manager.reference_smiles)
        self.data = {'rollouts': None, 'formula_dfs': None, 'formula_metrics': None, 'global_metrics': None}
        return self

    def evaluate(self, args: dict):
        self.data['rollouts'] = self.rollout_manager.perform_rollout(args['ac'], args['rollouts_from_file'], self.io_handler)
        self.data['formula_dfs'] = self.feature_calculator.calculate_features(self.data['rollouts'], self.io_handler, args['features_from_file'])
        self.data['formula_metrics'] = self.metrics_calculator.calculate_formula_metrics(self.data['formula_dfs'], self.io_handler)
        self.data['global_metrics'] = self.metrics_calculator.calculate_global_metrics(self.data['formula_metrics'])

        if self.wandb_run:
            self.data['global_metrics'].update({'total_num_steps': args['total_num_steps']})
            self.wandb_run.log(self.data['global_metrics'])

