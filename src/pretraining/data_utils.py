
import os, glob, re, pickle

import pandas as pd
import numpy as np

from ase import Atoms
from xtb.ase.calculator import XTB

from src.pretraining.pickle_to_h5 import load_data_from_hdf5


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


def find_files_with_prefix(data_dir, prefix, extension):
    pattern = os.path.join(data_dir, f"{prefix}_*.{extension}")
    paths = glob.glob(pattern)
    paths = [path for path in paths if not re.search('eval', path)]
    return paths, len(paths)


def load_data_generator(file_paths):
    for file_path in file_paths:
        if file_path.split('.')[-1] == 'pkl':
            with open(file_path, 'rb') as f:
                file_data = pickle.load(f)
                yield file_data
        elif file_path.split('.')[-1] == 'h5':
            yield load_data_from_hdf5(file_path)



def convert_ev_to_hartree(energy):
    return energy/27.2107


def load_dataset(mol_dataset: str) -> pd.DataFrame:

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_name = 'tmQM' if mol_dataset == 'TMQM' else 'qm7_E_smiles_connectivity'
    path_name = f'../../../TMQM-dataset/{mol_dataset.lower()}/processed/{dataset_name}.pkl'
    with open(os.path.join(dir_path, path_name), 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')

    return df



def find_best_molecule(row_ids: list, df: pd.DataFrame):
    best_energy = np.inf

    for i in row_ids:
        row = df.loc[i]
        if "energy_GFN2" not in row:
            atoms = Atoms(row['atomic_symbols'], row['pos'])
            atoms.calc = XTB(method='GFN2-xTB')
            energy = convert_ev_to_hartree(atoms.get_potential_energy())
        else:
            energy = row['energy_GFN2']

        if energy < best_energy:
            best_energy = energy
            # best_atoms = atoms
    
    return best_energy



def get_benchmark_energies(bags: list, mol_dataset: str, calculate_return=False):
    # TODO: finish this function

    df = load_dataset(mol_dataset)

    benchmark_energies = []
    benchmark_returns = [] if calculate_return else None

    for bag in bags:
        row_ids = df.index[df['bag_repr'] == bag]

        if len(row_ids) == 0:
            print(f'Bag {bag} not found in dataset. Skipping...')
            benchmark_energies.append(None)
            continue
        if len(row_ids) == 1:
            row_ids = row_ids.tolist()

        if df.loc[row_ids[0]]["n_atoms"] > 30:
            print(f'Bag {bag} has more than 30 atoms. Skipping...')
            benchmark_energies.append(None)
            continue

        best_energy = find_best_molecule(row_ids, df)
        benchmark_energies.append(best_energy)

        if calculate_return:

            benchmark_returns.append()

    return benchmark_energies, benchmark_returns


def get_eval_formulas(mol_dataset):
    """From preprocessed data"""

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, f'../../pretrain_runs/expert_data/{mol_dataset}_eval_bags.pkl'), 'rb') as f:
        eval_formulas = pickle.load(f)
    eval_formulas = sorted(list(set(eval_formulas["eval_bags"])))
    benchmark_energies, _ = get_benchmark_energies(eval_formulas, mol_dataset)
    eval_formulas, benchmark_energies = zip(*[(f, e) for f, e in zip(eval_formulas, benchmark_energies) if e is not None])

    print(f'Loaded {len(eval_formulas)} eval formulas for {mol_dataset} dataset. \nFormulas: {eval_formulas}')
    print(f'len eval formulas: {len(eval_formulas)}')
    print(f'number of unique eval formulas: {len(set(eval_formulas))}')

    return eval_formulas, benchmark_energies





from src.tools import util

def get_pretraining_formulas(n_atoms_max: int=20, n_test: int=100) -> list:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../../TMQM-dataset/tmQM.pkl'), 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame.from_dict(data, orient='index').sort_values(by='n_atoms')
    df = df[df['n_atoms'] <= n_atoms_max]
    TMs_keep = ['Ti', 'Zr', 'V', 'Cr', 'Mo', 'W', 'Mn', 'Re', 'Fe', 'Ru', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Cu', 'Zn'] # , 'Ag', 'Au'] 
    print(f'Keeping {len(TMs_keep)} transition metals: {TMs_keep}')
    # remove other specific symbols
    # forbidden = ['Ir']
    # TMs_keep = [x for x in TMs_keep if x not in forbidden]
    print(f'Keeping {len(TMs_keep)} transition metals: {TMs_keep}')

    df = df[df['symbols_sorted'].apply(lambda x: any(item in x for item in TMs_keep))]
    df['formulas'] = df['bag_repr'].apply(lambda x: util.string_to_formula(x))

    # Split df into train, val, test
    df_test = df.sample(n_test)
    df_train = df.drop(df_test.index)

    # TODO: Do some clever stratification to make sure the same TM elements are present in both train and test

    zs = [0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, \
    34, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 53, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80] # all unique tmqm elements  
    canvas_size = 75

    return df_train, df_test, zs, canvas_size



def get_bag_df(df: pd.DataFrame):
    """ Extract dataframe of unique bags """
    bag_df = df['bag_repr'].value_counts().rename('isomers').to_frame().reset_index().rename(columns={'index': 'bag_repr'})
    bag_df['n_atoms'] = bag_df.bag_repr.map(df.groupby('bag_repr')['n_atoms'].mean()).astype(int)
    return bag_df


def get_pretraining_formulas_QMx(n_atoms_min: int = 10, n_atoms_max: int=20, n_test: int=100, mol_dataset: str='QM7'):
    if mol_dataset == 'QM7':
        dataset_name = 'qm7' 
    elif mol_dataset == 'QM9':
        dataset_name = 'qm9'
    else:
        raise ValueError("Only QM7 and QM9 supported for now")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, f'../../../TMQM-dataset/{dataset_name}/processed/{dataset_name}_E_smiles_connectivity.pkl'), 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame.from_dict(data, orient='index').sort_values(by='n_atoms')

    df = df[~df['SMILES'].isnull()]
    
    #df = df[df['n_atoms'] <= n_atoms_max]
    df['formulas'] = df['bag_repr'].apply(lambda x: util.string_to_formula(x))

    # Split df into train, val, test
    # eval_bags = df['bag_repr'].value_counts().tail(n_test).index.values.tolist()

    eval_isomer_count = df[(df['n_atoms'] >= n_atoms_min) & (df['n_atoms'] <= n_atoms_max)]['bag_repr'].value_counts().tail(n_test)
    eval_formulas = eval_isomer_count.index.values.tolist()

    df_train = df[~df['bag_repr'].isin(eval_formulas)]

    eval_smiles = {
        f: df[(df['bag_repr'] == f) & (~df['SMILES'].isnull())]['SMILES'].values.tolist() 
        for f in eval_formulas
    }
    # print(f"eval_smiles: {eval_smiles}")
    # print(f"sum of eval_smiles: {sum([len(v) for v in eval_smiles.values()])}")
    # exit()

    eval_isomer_count = eval_isomer_count.to_dict()
    return df_train, eval_formulas, eval_smiles



def get_QM7_enhanced_bags():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, '../../../TMQM-dataset/bags_enhanced_qm7.pkl'), 'rb') as f:
        bag_list = pickle.load(f)
    return bag_list


from copy import deepcopy
import logging
def get_train_and_eval_data(config: dict, n_atoms_min: int = 10, n_atoms_max: int=20, n_test: int=100):
    # Bags/formulas
    if config['mol_dataset'] == 'QM7' or config['mol_dataset'] == 'QM9':
        zs, canvas_size = ([0, 1, 6, 7, 8, 16], 23) if config['mol_dataset'] == 'QM7' else ([0, 1, 6, 7, 8, 9], 29)
        if config['formulas'] is not None:
            train_formulas = util.split_formula_strings(config['formulas'])
            if config['eval_on_train']:
                config['eval_formulas'] = config['formulas']
                eval_formulas = util.split_formula_strings(config['eval_formulas'])
            else:
                raise ValueError("TODO: get_pretraining_formulas_QM7() to take training_formulas as input, \
                                 such that eval_formulas will be chosen from all minus train set")
        else:
            df_train, eval_formulas, eval_smiles = get_pretraining_formulas_QMx(n_atoms_min=n_atoms_min, 
                                                                                n_atoms_max=n_atoms_max, 
                                                                                n_test=n_test,
                                                                                mol_dataset=config['mol_dataset'])
            train_formulas = sorted(list(set(df_train['bag_repr'].values.tolist())))
            if config['eval_on_train']:
                eval_formulas = deepcopy(train_formulas) # override eval_formulas just calculated
    elif config['mol_dataset'] == 'QM7_enhanced':
        zs = [0, 1, 6, 7, 8, 16]
        canvas_size = 23
        _, eval_formulas, _ = get_pretraining_formulas_QMx(n_atoms_min=n_atoms_min,
                                                        n_atoms_max=n_atoms_max,
                                                        n_test=n_test,
                                                        mol_dataset=config['mol_dataset'])
        training_bags = get_QM7_enhanced_bags() # bag_tuples
        sizes = [util.get_formula_size(bag_tuple) for bag_tuple in training_bags]
        canvas_size = max(sizes) if max(sizes) > canvas_size else canvas_size
        train_formulas = [util.bag_tuple_to_str_formula(bag_tuple) for bag_tuple in training_bags]
        train_formulas = [f for f in train_formulas if f not in eval_formulas]
    else:
        raise ValueError("Only QM7 supported for now")

    if (config['train_mode'] == 'singlebag' or config['train_mode'] == 'finetuning') and len(train_formulas) == 1:
        assert eval_formulas == train_formulas

    if config['mol_dataset'] == 'QM7_enhanced':
        logging.info(f'Training on {len(train_formulas)} bags from QM7_enhanced')
        logging.info(f'First 100: {train_formulas[:100]}')
    else: 
        logging.info(f'Training bags: {train_formulas}')
    logging.info(f'Evaluation bags: {eval_formulas}')
    
    benchmark_energies, _ = get_benchmark_energies(eval_formulas, config['mol_dataset'])
    
    return zs, canvas_size, eval_formulas, train_formulas, df_train, benchmark_energies, eval_smiles



def get_partial_canvas_data(config: dict, n_atoms_min: int = 10, n_atoms_max: int=20, n_test: int=1):
    zs, canvas_size = ([0, 1, 6, 7, 8, 16], 23) if config['mol_dataset'] == 'QM7' else ([0, 1, 6, 7, 8, 9], 29)
    df = load_dataset(config['mol_dataset'])
    df['formulas'] = df['bag_repr'].apply(lambda x: util.string_to_formula(x))

    df['con_mat'] = df.apply(
        lambda x: unpack_upper_triangular_to_full(x['con_upper_triangular'], x['n_atoms'])
        , axis=1
    )
    # df['con_mat'] = df['con_mat'].apply(lambda x: x.astype(np.int8))
    df = df.drop(columns=['con_upper_triangular'])

    # Filter out mols that couldn't be converted to RdKit mol object
    print(f"SMILES == None for : {len(df[df['SMILES'].isnull()])} / {len(df)}")
    df = df[~df['SMILES'].isnull()]

    # We can now select the subset of molecules that we want to use for canvas generation during training and eval.
    split_method = 'formula_statified' # 'random' or 'formula_statified'

    if split_method == 'random':
        # Select n_test eval molecules randomly
        df_eval = df.sample(n_test)
        df_train = df.drop(df_eval.index)
        
    elif split_method == 'formula_statified':
        # Select subset viable for evaluation (n_atoms_min <= n_atoms <= n_atoms_max)
        df_filtered = df[(df['n_atoms'] >= n_atoms_min) & (df['n_atoms'] <= n_atoms_max)]

        # We sort them by isomer count to keep as many structures in the training set as possible
        #eval_isomer_count = df_filtered['bag_repr'].value_counts().tail(n_test) # pandas.core.series.Series
        # print(df_filtered['bag_repr'].value_counts())

        eval_isomer_count = df_filtered['bag_repr'].value_counts().sample(n_test) # pandas.core.series.Series
        # print(eval_isomer_count)

        # Sorting by isomer count yields highly unsaturated molecules. We want rings also.
        eval_formulas = pd.DataFrame(eval_isomer_count).index.values.tolist()

        # Create train and eval sets
        df_eval = df_filtered[df_filtered['bag_repr'].isin(eval_formulas)]
        df_train = df[~df['bag_repr'].isin(eval_formulas)]
    

    return zs, canvas_size, df_train, df_eval, eval_formulas
    
