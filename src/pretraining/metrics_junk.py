
import pickle

from typing import List, Optional, Union, Dict
from io import StringIO 

import pandas as pd
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


from src.performance.xyz2mol import xyz2mol


def validate_sanitize(mol):
    # https://www.programcreek.com/python/example/96360/rdkit.Chem.SanitizeMol
    # Seems to sometimes accept invalid molecules
    try:
        Chem.SanitizeMol(
            mol,
            # Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            # | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            # | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            # | Chem.SanitizeFlags.SANITIZE_CLEANUP,
        )
        if mol:
            return True
    except:
        return False


def validate_sanitize_charge_loop(atoms: Atoms):
    is_valid = False
    charges = [-2, -1, 0, 1, 2]
    if mol_with_inferred_bonds(atoms, try_charges=charges):
        is_valid = True
    return is_valid


def mol_with_inferred_bonds(
    atoms: Atoms, try_charges: Optional[list[int]] = None
) -> Union[Chem.Mol, None]:
    if try_charges is None:
        try_charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    xyz_str = atoms_to_xyz_str(atoms)
    return_mol = None

    raw_mol = Chem.MolFromXYZBlock(xyz_str)
    if raw_mol is not None:
        for charge in try_charges:
            try:
                mol = Chem.Mol(raw_mol)
                rdDetermineBonds.DetermineBonds(
                    mol,
                    charge=charge,
                )
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    mol,
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                    | Chem.SanitizeFlags.SANITIZE_CLEANUP,
                )
                if mol:
                    return mol
                else:
                    continue
            except ValueError:
                continue
    return return_mol


def atoms_to_xyz_str(atoms: Atoms):
    f = StringIO()
    atoms.write(f, format="xyz")
    return f.getvalue()



# def atom_list_to_df(atoms_object_list: List[Atoms] = None):

#     feature_cols = ['SMILES', 'valid', 'abs_energy', 'NEW_SMILES', 'relax_stable', 'basin_distance', 'RMSD']
#     df = pd.DataFrame(columns=feature_cols)

#     for i in range(len(atoms_object_list)):
#         atoms = atoms_object_list[i]

#         mol_info = get_mol(atoms)
#         valid = mol_info['info'] == 'valid'
#         mol = mol_info['mol']

#         abs_energy = calc_potential_energy(atoms)
#         SMILES = Chem.MolToSmiles(mol) if mol else None


#         perform_optimization = False
#         if valid and perform_optimization:
#             opt_info = optimize_atoms(atoms, max_steps=200, fmax=0.10, method='GFN2-xTB')
#             new_mol_list = get_mol(opt_info['new_atoms'])
#             new_mol = new_mol_list[0] if new_mol_list else None

#             NEW_SMILES = Chem.MolToSmiles(new_mol) if new_mol else None
#             relax_stable = True if NEW_SMILES == SMILES else False

#             if valid and relax_stable and opt_info['converged']:
#                 basin_distance = abs(opt_info['energy_after'] - opt_info['energy_before']) 
#                 RMSD = Chem.rdMolAlign.GetBestRMS(mol, new_mol)
#             else:
#                 basin_distance = None
#                 RMSD = None
#         else:
#             NEW_SMILES = None
#             relax_stable = None
#             basin_distance = None
#             RMSD = None

#         df = pd.concat([df, pd.DataFrame(
#             [
#                 [
#                     SMILES,
#                     valid,
#                     abs_energy,
#                     #reward,
#                     NEW_SMILES,
#                     relax_stable,
#                     basin_distance,
#                     RMSD
#                 ]
#             ], columns=feature_cols)]
#         )
    
#     return df


# def atom_list_to_df(atoms_object_list: List[Atoms] = None):

#     # Define a dictionary that maps feature names to their data types
#     features_dict = {
#         'valid': bool,
#         'SMILES': str,
#         'SMILES_Compact': str,
#         'n_rings': int,
#         'charge_fail': bool,
#         'abs_energy': float,
#         'NEW_SMILES': str,
#         'relax_stable': bool,
#         'basin_distance': float,
#         'RMSD': float
#     }

#     # Create an empty DataFrame with the correct column names
#     feature_cols = list(features_dict.keys())
#     df = pd.DataFrame(columns=feature_cols)

#     # Apply the data types from the dictionary to the columns
#     for col, dtype in features_dict.items():
#         df[col] = df[col].astype(dtype)

#     for i in range(len(atoms_object_list)):
#         atoms = atoms_object_list[i]

#         mol_info = get_mol(atoms)
#         valid = mol_info['info'] == 'valid'
#         mol = mol_info['mol']

#         charge_fail = True if mol_info['info'] == 'charged_or_radical' else False
#         n_rings = mol.GetRingInfo().NumRings() if mol else None


#         abs_energy = calc_potential_energy(atoms)
#         SMILES = Chem.MolToSmiles(mol) if mol else None
#         mol_pos_free = Chem.MolFromSmiles(SMILES) if SMILES is not None else None
#         SMILES_Compact = Chem.MolToSmiles(mol_pos_free, isomericSmiles=True) if mol_pos_free else None

#         perform_optimization = True
#         if valid and perform_optimization:
#             opt_info = optimize_atoms(atoms, max_steps=200, fmax=0.10, method='GFN2-xTB')

#             new_mol_info = get_mol(opt_info['new_atoms'])
#             new_valid = new_mol_info['info'] == 'valid'
#             new_mol = new_mol_info['mol']

#             NEW_SMILES = Chem.MolToSmiles(new_mol) if new_mol else None
#             relax_stable = True if new_valid and NEW_SMILES == SMILES else False

#             if valid and relax_stable and opt_info['converged']:
#                 basin_distance = abs(opt_info['energy_after'] - opt_info['energy_before']) 
#                 RMSD = Chem.rdMolAlign.GetBestRMS(mol, new_mol)
#             else:
#                 basin_distance = None
#                 RMSD = None
#         else:
#             NEW_SMILES = None
#             relax_stable = None
#             basin_distance = None
#             RMSD = None

#         df = pd.concat([df, pd.DataFrame(
#             [
#                 [
#                     valid,
#                     SMILES,
#                     SMILES_Compact,
#                     n_rings,
#                     charge_fail,
#                     abs_energy,
#                     #reward,
#                     NEW_SMILES,
#                     relax_stable,
#                     basin_distance,
#                     RMSD
#                 ]
#             ], columns=feature_cols)]
#         )
    
#     return df


def calculate_aggregate_metrics(df: pd.DataFrame, SMILES_db: list = None):
    agg_metrics = {}
    unique_smiles = [x for x in df['SMILES'].unique() if str(x) != 'nan']
    SMILES_db = list(set([x for x in SMILES_db if str(x) != 'nan'])) if SMILES_db is not None else None


    # Validity ratios
    agg_metrics['valid_per_sample'] = sum(df['valid']) / len(df) if len(df) > 0 else 0
    # agg_metrics['valid_per_attempt'] =  sum(df['valid']) / (len(df) + killed)

    # Diversity measures
    agg_metrics['total_mols'] = df['SMILES'].nunique()

    # Uniqueness measures
    agg_metrics['mols_per_sample'] = df['SMILES'].nunique() / len(df) if len(df) > 0 else 0
    agg_metrics['mols_per_valid'] = df['SMILES'].nunique() / sum(df['valid']) if sum(df['valid']) > 0 else 0

    # Rediscovery measures
    if SMILES_db:
        agg_metrics['rediscovered'] = sum([1 if sm in SMILES_db else 0
                                            for sm in unique_smiles]) if SMILES_db else 0
        agg_metrics['rediscovery_ratio'] = sum([1 if sm in unique_smiles else 0 
                                            for sm in SMILES_db]) / len(SMILES_db) if SMILES_db else 0
        agg_metrics['expansion_ratio'] = sum([1 if sm_rl not in SMILES_db else 0 
                                            for sm_rl in unique_smiles]) / len(SMILES_db) if SMILES_db else 0



    # Energy measures
    agg_metrics['abs_energy_avg'] = df['abs_energy'].mean()

    # After relaxation measures
    agg_metrics["relax_stable"] = np.sum(df["relax_stable"]) / len(df) if len(df) > 0 else 0
    agg_metrics['RMSD_avg'] = df['RMSD'].mean()

    # Ring measures
    agg_metrics['n_rings_avg'] = df['n_rings'].mean()
    #agg_metrics['ring4+'] = sum(df['n_rings'] >= 4) / len(df) if len(df) > 0 else 0
    #agg_metrics['ring5+'] = sum(df['n_rings'] >= 5) / len(df) if len(df) > 0 else 0

    return agg_metrics

def calculate_aggregate_metrics2(df: pd.DataFrame):
    agg_metrics = {}

    # Diversity measures
    agg_metrics['total_mols'] = df['SMILES_Compact'].nunique()
    agg_metrics['mols_per_sample'] = df['SMILES_Compact'].nunique() / len(df) if len(df) > 0 else 0
    agg_metrics['mols_per_valid'] = df['SMILES_Compact'].nunique() / sum(df['valid']) if sum(df['valid']) > 0 else 0

    return agg_metrics




class Metrics:
    """ Not yet in use """

    def __init__(self):
        self.feature_cols = ['parse_mol', 'valid', 'abs_energy', 'basin_distance', 'SMILES', 'SMILES_CANONICAL', 'SMILES_CANONICAL2']
        self.metric_functions = {
            'parse_mol': validate_parse_mol_xyz2mol,
            'abs_energy': calc_potential_energy,
            'basin_distance': basin_dist,
            'valid': validate_sanitize,
            'SMILES': lambda mol: Chem.MolToSmiles(mol) if mol else None,
            'SMILES_CANONICAL': lambda mol: Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None,
            'SMILES_CANONICAL2': lambda mol: Chem.MolToSmiles(mol, canonical=True) if mol else None,
        }

    def calculate_metric(self, metric_name, atoms):
        metric_function = self.metric_functions.get(metric_name)
        if metric_function:
            return metric_function(atoms)
        return None

    def atom_list_to_df(self, atoms_object_list: List[Atoms] = None, metrics: List[str] = None):
        selected_cols = [col for col in self.feature_cols if col in metrics]
        df = pd.DataFrame(columns=selected_cols)

        for atoms in atoms_object_list:
            metrics_row = []
            for metric_name in selected_cols:
                metric_value = self.calculate_metric(metric_name, atoms)
                metrics_row.append(metric_value)
            df = pd.concat([df, pd.DataFrame([metrics_row], columns=selected_cols)], ignore_index=True)

        return df
    




# def main():

#     # Test a TMQM molecule (SAJDEO)
#     CSD_code = 'SAJDEO'
#     with open('../TMQM-dataset/tmQM.pkl', 'rb') as f:
#         tmqm_dict = pickle.load(f)

#     molecule_dict = tmqm_dict[CSD_code]
#     atoms = Atoms(symbols=molecule_dict['atomic_symbols'], positions=molecule_dict['pos'])
#     validity_tmqm = get_mol(atoms)['info']
#     print(f'validity_tmqm: {validity_tmqm}. (zyx2mol only works for purely organic molecules)')


#     # Test a bunch of QM7 molecules. They should all be good
#     with open('../TMQM-dataset/qm7.pkl', 'rb') as f:
#         qm7_dict = pickle.load(f) 

#     all_atoms = []
#     all_validities = []
#     for mol_index in range(10):
#         mol_dict = qm7_dict[mol_index]
#         atoms = Atoms(symbols=mol_dict['atomic_symbols'], positions=mol_dict['pos'])
#         all_atoms.append(atoms)
#         all_validities.append(get_mol(atoms)['info'] == 'valid')

#     validity_dict = dict(zip(range(len(all_validities)), all_validities))
#     print(f'qm7 validity: {validity_dict}')


# if __name__ == '__main__':
#     main()