from rdkit import Chem
from typing import Union
from .Adduct import Adduct
from ..Mol.Formula import Formula
from ..Mol.Compound import Compound


# Suppress warnings and informational messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

class AdductIon:
    def __init__(self, compound: Compound, adduct: Adduct):
        self.compound = compound
        self.adduct = adduct

        pass

    def __repr__(self):
        return f"AdductedIon({self.compound.smiles} {self.adduct})"
    
    @property
    def formula(self) -> Formula:
        """
        Get the molecular formula of the fragment with the adduct applied.
        """
        formula = self.compound.formula.copy()
        formula = formula + self.adduct.formula
        formula._charge = self.adduct.charge
        return formula

    @property
    def charge(self) -> int:
        """
        Get the charge of the fragment with the adduct applied.
        """
        return self.adduct.charge
        
    @property
    def mz(self) -> float:
        """
        Get the mass-to-charge ratio of the fragment.
        """
        total_charge = self.charge
        total_mass = self.formula.exact_mass

        # assert total_charge != 0, "Charge must be non-zero to calculate m/z"
        if total_charge == 0:
            raise ValueError("Charge must be non-zero to calculate m/z")

        return total_mass / abs(total_charge)
    
    