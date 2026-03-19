import time
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


from ase import Atoms
from ase.io import read
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tools import util
from src.data.io_handler import IOHandler
from src.performance.energetics import XTBOptimizer, EnergyUnit
from src.performance.reward_metrics_rings import get_max_view_positions



import subprocess
import json
import shlex
import argparse

def _optional_int(arg: str):
    if arg is None:
        return None
    if isinstance(arg, str) and arg.lower() in {"none", "null"}:
        return None
    return int(arg)

def submit_job(params, script_path: str="scripts/analyse/exp5_scaffold/visuals/chimerax_script.py"):

    # Convert dictionary to a JSON string
    params_json = json.dumps(params)
    quoted_params_json = shlex.quote(params_json)

    print(f"quoted_params_json: {quoted_params_json}")

    # Construct the command correctly
    cmd = [
        "chimerax", "--nogui", "--offscreen",
        "--script", f'"{script_path} {quoted_params_json}"'  # Properly quote the JSON
    ]

    print("Executing:", " ".join(cmd))  # Debugging
    subprocess.run(" ".join(cmd), shell=True, check=True)  # Run as a single shell command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sampled molecules with ChimeraX.")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=None,
        help=(
            "One or more run directories (e.g. pretrain_runs/A-30k-Fixed/seed_0). "
            "Expected layout: <run_dir>/results/<tag>/<formula>/{df.csv,atoms.traj}"
        ),
    )
    parser.add_argument("--base_dir", type=str, default=None, help="Optional base directory (used with --run_names/--n_seeds).")
    parser.add_argument("--run_names", nargs="+", default=None, help="Optional run names (used with --base_dir/--n_seeds).")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds (used with --base_dir/--run_names).")

    parser.add_argument("--tag", type=str, default="exp5-31500", help="Results tag under results/<tag>/")
    parser.add_argument("--eval_formulas", nargs="+", default=None, help="List of formula strings to visualize.")

    parser.add_argument(
        "--n_query",
        type=_optional_int,
        default=None,
        help="Number of molecules matching ring query. Use 'None' to disable stratification.",
    )
    parser.add_argument(
        "--n_non_query",
        type=_optional_int,
        default=None,
        help="Number of molecules not matching ring query. Use 'None' to disable stratification.",
    )

    parser.add_argument("--sorting_key", type=str, default="dipole", help="Column used to sort df.csv.")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="SMILES column name in df.csv.")
    parser.add_argument("--stratify_on_smiles", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--bg_color_str", type=str, default="black", help="ChimeraX background color.")
    parser.add_argument("--show_relaxed", action=argparse.BooleanOptionalAction, default=True, help="Relax molecules and sort by relaxed dipole.")
    parser.add_argument("--illustration_dir_name", type=str, default=None, help="Output dir name under each run_dir (defaults to exp5_<bg>_<sorting_key>).")
    return parser.parse_args()


def default_eval_formulas() -> List[str]:
    return [
        'H4C3O3',  # Bad
        'H6C3O3',  # Good
        'H6C4O3',  # Bad
        'H8C4O3',  # Good
        'H6C5O3',  # Bad
        'H8C5O3',  # Bad
        'H10C5O3', # Good
    ]


def build_run_dirs(args: argparse.Namespace) -> List[str]:
    if args.run_dirs:
        return args.run_dirs
    if not (args.base_dir and args.run_names):
        raise SystemExit("Provide either --run_dirs or (--base_dir and --run_names).")
    run_dirs: List[str] = []
    for run_name in args.run_names:
        for seed in range(args.n_seeds):
            run_dirs.append(str(Path(args.base_dir) / run_name / f"seed_{seed}"))
    return run_dirs


def get_eval_path(run_dir: str, tag: str, formula: str) -> Path:
    return Path(run_dir) / "results" / tag / formula

def get_all_dfs(eval_formulas: List[str], run_dirs: List[str], tag: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Atoms]]]:
    dfs = {}
    trajs = {}

    for formula in tqdm(eval_formulas, desc='Loading data'):
        dfs[formula] = []
        trajs[formula] = []

        paths = [get_eval_path(run_dir, tag, formula) for run_dir in run_dirs]
        existing_paths = [path for path in paths if path.exists()]

        if not existing_paths:
            print(f"Missing all paths for formula={formula} tag={tag}")
            continue

        for seed_path in tqdm(existing_paths, desc='Loading data'):
            # load df
            df = pd.read_csv(os.path.join(seed_path, 'df.csv'))
            dfs[formula].append(df)


            # load atoms.traj
            atoms_traj_path = os.path.join(seed_path, 'atoms.traj')
            atoms_list = read(atoms_traj_path, index=':')
            trajs[formula].extend(atoms_list)
            #break
        #break

    valid_formulas = [formula for formula in eval_formulas if len(dfs.get(formula, [])) > 0]

    if not valid_formulas:
        print("No valid dataframes found for any formulas. Skipping.")
        return {}, {}

    all_dfs = {formula: pd.concat(dfs[formula]).reset_index(drop=True) for formula in valid_formulas}
    trajs = {formula: trajs[formula] for formula in valid_formulas}

    return all_dfs, trajs


def filter_data(
    all_dfs: Dict[str, pd.DataFrame],
    trajs: Dict[str, List[Atoms]],
    eval_formulas: List[str],
    sorting_key: str,
    smiles_col: str,
    stratify_on_smiles: bool
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Atoms]]]:

    merged_dfs = all_dfs.copy()
    merged_trajs = trajs.copy()

    trajs_filtered = {}
    dfs_filtered = {}


    for formula in eval_formulas:
        df = merged_dfs[formula].copy().reset_index(drop=True)
        # Sort by sorting_key (descending: highest dipole / best energy first)
        df = df.sort_values(by=sorting_key, ascending=False)
        
        # Drop rows where SMILES is None
        df = df[df[smiles_col].notna()]

        # Drop rows where relax_stable is False
        # df = df[df['relax_stable']]

        # Keep only the best of each SMILES
        if stratify_on_smiles:
            df = df.drop_duplicates(subset=[smiles_col], keep='first')

        # Update the trajectories
        idx = df.index.copy()
        trajs_filtered[formula] = [merged_trajs[formula][i] for i in idx]
        dfs_filtered[formula] = df.reset_index(drop=True)
    
    return dfs_filtered, trajs_filtered


def find_candidate_mols(
    dfs_filtered: Dict[str, pd.DataFrame],
    trajs_filtered: Dict[str, List[Atoms]],
    eval_formulas: List[str],
    n_query: int,
    n_non_query: int
) -> Dict[str, List[Atoms]]:
    """ Select the most promising molecules. """
    best_mols = {}


    for formula in eval_formulas:

        if n_query is None and n_non_query is None:
            print(f"No stratification on smiles, using all molecules for {formula}")
            best_mols[formula] = trajs_filtered[formula].copy()
            continue
        else:
            # Select molecules with rings and more than 4 atoms in the largest ring
            query = 'n_rings > 0 and n_atoms_ring_max > 4'
            queried_idx = dfs_filtered[formula].query(query).index
            queried_mols = [trajs_filtered[formula][i] for i in queried_idx[:n_query]].copy()
            
            # Other promising molecules based on energy (complementary to query)
            non_query_idx = dfs_filtered[formula].index.difference(queried_idx)
            non_query_mols = [trajs_filtered[formula][i] for i in non_query_idx[:n_non_query]].copy()

            # Combine the two lists
            best_mols[formula] = queried_mols + non_query_mols

    return best_mols




def optimize_and_sort_mols(
    best_mols: Dict[str, List[Atoms]],
    eval_formulas: List[str],
) -> Tuple[Dict[str, List[Atoms]], Dict[str, List[float]]]:
    calc = XTBOptimizer(method='GFN2-xTB', energy_unit=EnergyUnit.EV, use_mp=False)

    dipoles_relaxed = {}
    mols_relaxed = {}
    for formula in tqdm(eval_formulas, desc='Optimizing and sorting molecules'):
        print(f"Optimizing and sorting molecules for {formula}")
        dipoles = []
        m_relaxed = []
        for mol in tqdm(best_mols[formula], desc='Relaxing molecules'):
            opt_info = calc.optimize_atoms(mol, max_steps=1000, fmax=0.05, redirect_logfile=False)
            relaxed_mol = opt_info['new_atoms']
            # Calculate dipole moment for the optimized molecule
            dipole = calc.calc_dipole(relaxed_mol)
            # Use 0.0 if dipole calculation failed (None)
            dipole = dipole if dipole is not None else 0.0
            dipoles.append(dipole)
            m_relaxed.append(relaxed_mol)

        if not m_relaxed:
            print(f"No valid molecules for {formula}; skipping.")
            continue

        # Sort based on dipole moment (largest first)
        dipoles, m_relaxed = zip(*sorted(zip(dipoles, m_relaxed), key=lambda pair: pair[0], reverse=True))
        dipoles_relaxed[formula] = list(dipoles)
        mols_relaxed[formula] = list(m_relaxed)
        mols_to_view = mols_relaxed[formula]

    return mols_relaxed, dipoles_relaxed



def rotate_mols(mols: Dict[str, List[Atoms]]) -> Dict[str, List[Atoms]]:
    """ Rotate of the molecule to get a better view. """
    for formula, mols_to_view in mols.items():
        for mol in mols_to_view:
            mol.set_positions(get_max_view_positions(mol.get_positions()))



def launch_chimerax_jobs(mols_to_view: Dict[str, List[Atoms]], save_dir: str, bg_color_str: str):

    formulas = list(mols_to_view.keys())

    for formula in formulas:


        # Finally, write molecules to pdb and create Chimerax visualization (png).
        mol_paths = [os.path.join(save_dir, f'{formula}_{i}.pdb') for i in range(len(mols_to_view[formula]))]
        save_paths = [os.path.join(save_dir, f'{formula}_{i}.png') for i in range(len(mols_to_view[formula]))]

        for i, mol in enumerate(mols_to_view[formula]):
            mol.write(mol_paths[i])

        # Create Chimerax visualization (png).
        for i, (mol_path, save_path) in enumerate(zip(mol_paths, save_paths)):
            # Check if the PDB file exists before launching Chimerax
            if not os.path.exists(mol_path):
                print(f"Warning: PDB file {mol_path} does not exist, skipping Chimerax visualization")
                continue
            
            params = {
                "pdb_path": os.path.join(os.getcwd(), mol_path),
                "image_path": os.path.join(os.getcwd(), save_path),
                "movie_path": None,
                "atoms_to_deselect": [], # 0, 1, 2, 3]
                "selected_action": "",
                "bg_color": bg_color_str
            }

            print(f"params: {params}")
            submit_job(params)

    return


if __name__ == "__main__":

    args = parse_args()

    eval_formulas = args.eval_formulas if args.eval_formulas else default_eval_formulas()
    run_dirs = build_run_dirs(args)

    illustration_dir_name = args.illustration_dir_name or f"exp5_{args.bg_color_str}_{args.sorting_key}_{args.tag}"

    for run_dir in run_dirs:
        save_dir = str(Path(run_dir) / illustration_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        # Get all data
        start_time = time.time()
        all_dfs, trajs = get_all_dfs(eval_formulas, [run_dir], args.tag)
        print(f"a) Time taken to get all data: {time.time() - start_time} seconds")

        if not all_dfs:
            print(f"Skipping run_dir={run_dir} (no data found).")
            continue

        # Filter data
        start_time = time.time()
        dfs_filtered, trajs_filtered = filter_data(
            all_dfs,
            trajs,
            list(all_dfs.keys()),
            args.sorting_key,
            args.smiles_col,
            args.stratify_on_smiles,
        )
        print(f"b) Time taken to filter data: {time.time() - start_time} seconds")

        # Find candidate molecules
        start_time = time.time()
        best_mols = find_candidate_mols(
            dfs_filtered,
            trajs_filtered,
            list(dfs_filtered.keys()),
            args.n_query,
            args.n_non_query,
        )
        print(f"c) Time taken to find candidate molecules: {time.time() - start_time} seconds")

        # Optimize and sort molecules
        start_time = time.time()
        if args.show_relaxed:
            mols_to_view, _dipoles_relaxed = optimize_and_sort_mols(best_mols, list(best_mols.keys()))
        else:
            mols_to_view = best_mols
        print(f"d) Time taken to optimize and sort molecules: {time.time() - start_time} seconds")

        # Rotate molecules
        start_time = time.time()
        rotate_mols(mols_to_view)
        print(f"e) Time taken to rotate molecules: {time.time() - start_time} seconds")

        # Visualize molecules
        start_time = time.time()
        launch_chimerax_jobs(mols_to_view, save_dir, args.bg_color_str)
        print(f"f) Time taken to launch Chimerax jobs: {time.time() - start_time} seconds")
        
        
        
        

