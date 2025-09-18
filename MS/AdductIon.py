from rdkit import Chem
from typing import Union
from .Adduct import Adduct
from ..Mol.Formula import Formula
from ..Mol.Compound import Compound


# Suppress warnings and informational messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

class AdductIon:
    DELIM = "|"
    def __init__(self, compound: Compound, adduct: Adduct):
        self.compound = compound
        self.adduct = adduct

        pass

    def __repr__(self):
        return f"AdductIon({self.compound.smiles}{self.adduct})"
    
    def __str__(self) -> str:
        """
        Return string representation with delimiter.
        Example: "C6H12O6|[M+H]+"
        """
        return f"{self.compound.smiles}{self.DELIM}{self.adduct}"
    

    @classmethod
    def parse(cls, text: str) -> "AdductIon":
        """
        Parse string created by __str__ into an AdductIon.
        Example input: "C6H12O6|[M+H]+"
        """
        from ..Mol.Compound import Compound
        from .Adduct import Adduct

        if cls.DELIM not in text:
            raise ValueError(f"Missing delimiter '{cls.DELIM}' in: {text}")

        smiles, adduct_str = text.split(cls.DELIM, 1)

        # construct Compound and Adduct
        compound = Compound.from_smiles(smiles)  # factory: handles SMILES or Formula
        adduct = Adduct.parse(adduct_str)

        return cls(compound, adduct)
    
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
    
    