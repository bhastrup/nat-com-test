import logging
import numpy as np
from ase import Atoms
from xtb.ase.calculator import XTB
from ase.optimize import BFGS, FIRE
from enum import Enum


class EnergyUnit(Enum):
    HARTREE = 'hartree'
    KCAL_MOL = 'kcal/mol'
    EV = 'eV'


def str_to_EnergyUnit(value: str) -> EnergyUnit:
    """Converts a string to an EnergyUnit enum, case-insensitively."""
    value = value.lower()
    for unit in EnergyUnit:
        if unit.value.lower() == value:
            return unit
    raise ValueError(f"'{value}' is not a valid EnergyUnit")


class EnergyConverter:
    conversion_factors = {
        (EnergyUnit.HARTREE, EnergyUnit.KCAL_MOL): 627.50960,
        (EnergyUnit.HARTREE, EnergyUnit.EV): 27.21140,
        (EnergyUnit.KCAL_MOL, EnergyUnit.HARTREE): 1 / 627.50960,
        (EnergyUnit.EV, EnergyUnit.HARTREE): 1 / 27.21140,
        (EnergyUnit.KCAL_MOL, EnergyUnit.EV): 27.21140 / 627.50960,
        (EnergyUnit.EV, EnergyUnit.KCAL_MOL): 627.50960 / 27.21140,
    }
    
    @staticmethod
    def convert(values: np.ndarray, old_unit: EnergyUnit, new_unit: EnergyUnit) -> float:
        """Convert value from old_unit to new_unit."""
        if old_unit == new_unit:
            return values
        
        try:
            factor = EnergyConverter.conversion_factors[(old_unit, new_unit)]
            return values * factor
        except KeyError:
            raise ValueError(f"Unknown conversion from {old_unit} to {new_unit}")



import multiprocessing as mp
import logging

class XTBOptimizer:
    old_energy_unit = EnergyUnit.EV
    
    def __init__(
        self, 
        method: str = 'GFN2-xTB',
        energy_unit: EnergyUnit = EnergyUnit.HARTREE, 
        use_mp: bool = False,
        timeout: int = 60
    ):
        self.method = method
        self.energy_unit = energy_unit
        self.use_mp = use_mp
        self.timeout = timeout
        self.pool = mp.Pool(1) if use_mp else None # Create a persistent pool with 1 worker


    def get_max_force(self, atoms: Atoms) -> float:
        forces = atoms.get_forces()
        max_force = ((forces ** 2).sum(axis=1) ** 0.5).max()
        return EnergyConverter.convert(max_force, self.old_energy_unit, self.energy_unit)

    def _calculate_energy(self, atoms: Atoms) -> float:
        atoms = atoms.copy()
        atoms.calc = XTB(method=self.method)
        return atoms.get_potential_energy()

    def calc_potential_energy(self, atoms: Atoms) -> float:
        """ Wrapper around _calculate_energy() to handle mp, timeouts and energy unit conversion."""
        try:
            if self.use_mp:
                result = self.pool.apply_async(self._calculate_energy, (atoms,))
                energy = result.get(timeout=self.timeout)
            else:
                energy = self._calculate_energy(atoms)

            return EnergyConverter.convert(energy, self.old_energy_unit, self.energy_unit)
        except mp.TimeoutError:
            logging.error("XTB calculation timed out.")
            return None
        except Exception as e:
            logging.error(f"Error in XTB calculation: {e}")
            return None


    def optimize_atoms(self, atoms: Atoms, max_steps: int = 50, fmax: float = 0.10, redirect_logfile: bool = True) -> dict:
        atoms = atoms.copy()
        atoms.calc = XTB(method=self.method)

        energy_before = self.calc_potential_energy(atoms)

        if energy_before is not None:
            try:
                dyn = FIRE(atoms, logfile='temp_ASE_optimizer_log_file' if redirect_logfile else None)  # dyn = BFGS(atoms)
                dyn.run(fmax=fmax, steps=max_steps)
                energy_after = self.calc_potential_energy(atoms)
            except Exception as e:
                logging.error(f"Error during optimization: {e}")
                energy_after = None
        else:
            energy_after = None

        try:
            fmax_final = self.get_max_force(atoms)
        except Exception as e:
            logging.error(f"Error in calculating max force: {e}")
            fmax_final = fmax

        converged = fmax_final < fmax

        atoms.calc = None

        opt_info = dict(
            energy_before=energy_before,
            energy_after=energy_after,
            new_atoms=atoms,
            fmax_final=fmax_final,
            converged=converged,
            nsteps=dyn.nsteps if converged else max_steps
        )

        return opt_info

    def close_pool(self):
        """Clean up the multiprocessing pool."""
        self.pool.close()
        self.pool.join()
