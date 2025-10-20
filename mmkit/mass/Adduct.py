from typing import Dict, List, OrderedDict, Literal
import re
from rdkit import Chem
from collections import defaultdict
from ..chem.Formula import Formula
from ..chem.utilities import charge_from_str

class Adduct:
    """
    Class representing an adduct ion.
    """
    def __init__(
            self, 
            ion_type: Literal["M", "F"], 
            n_molecules: int = 1,
            adducts_in: List[Formula] = [], 
            adducts_out: List[Formula] = [], 
            charge_offset: int = 0
            ):
        """
        Initialize an adduct with element differences and charge difference.
        
        Args:
            ion_type (str): ion_type of the adduct, either "M" or "F". 
            element_diff (Dict[str, int]): Dictionary of element differences.
            charge_offset (int): Additional charge offset to apply.
        """
        assert ion_type in ["M", "F"], f"ion_type must be one of 'M' or 'F', but got '{ion_type}'."
        assert n_molecules >= 1, "n_molecules must be at least 1"

        self._ion_type = ion_type
        self._n_molecules = n_molecules
        formula_count_in: dict[Formula, int] = defaultdict(int)
        charge = 0
        for f in adducts_in:
            formula_count_in[f] += 1
            charge += f.charge
        for f in adducts_out:
            formula_count_in[f] -= 1
            charge -= f.charge

        self._adduct_formulas: dict[Formula, int] = {f.copy(): cnt for f, cnt in formula_count_in.items()}
        self._charge = charge + charge_offset

    @property
    def formula_shift(self) -> Formula:
        """
        Get the combined formula of the adduct.
        
        Returns:
            Formula: Combined formula of the adduct.
        """
        formula = Formula.empty()
        for f, cnt in self._adduct_formulas.items():
            formula = formula + (f * cnt)
        return formula

    @property
    def mass_shift(self) -> float:
        """
        Get the mass shift of the adduct.
        
        Returns:
            float: Mass shift of the adduct.
        """
        total_mass = sum(f.exact_mass * cnt for f, cnt in self._adduct_formulas.items() if cnt != 0)
        return total_mass
    
    @property
    def charge(self) -> int:
        """
        Get the charge of the adduct.
        
        Returns:
            int: Charge of the adduct.
        """
        return self._charge
    
    def set_charge(self, charge: int):
        """
        Set the charge of the adduct.
        
        Args:
            charge (int): New charge value.
        """
        self._charge = charge
    
    def __repr__(self) -> str:
        """
        Get the string representation of the adduct.
        
        Returns:
            str: String representation of the adduct.
        """
        return f'Adduct({self.__str__()})'

    def __str__(self) -> str:
        """
        Get the string representation of the adduct.
        
        Returns:
            str: String representation of the adduct.
        """
        parts = []
        for f, cnt in sorted(self._adduct_formulas.items(), key=lambda x: (-x[0].exact_mass, x[0].raw_formula)):
            if cnt > 0:
                parts.append(f"+{cnt if cnt > 1 else ''}{f.raw_formula.replace('+', '').replace('-', '')}")
            elif cnt < 0:
                parts.append(f"-{abs(cnt) if cnt < -1 else ''}{f.raw_formula.replace('+', '').replace('-', '')}")
        body = "".join(parts)

        if self._charge > 0:
            charge = f"+{self._charge}" if self._charge > 1 else "+"
        elif self._charge < 0:
            charge = f"{self._charge}" if self._charge < -1 else "-"
        else:
            charge = ""
            
        nM = f"{self._n_molecules if self._n_molecules > 1 else ''}{self._ion_type}"

        return f"[{nM}{body}]{charge}"
    
    def __eq__(self, value):
        return self.__str__() == str(value)
    
    def __hash__(self):
        """
        Get the hash of the adduct.
        
        Returns:
            int: Hash of the adduct.
        """
        return hash(self.__str__())
    
    def copy(self) -> "Adduct":
        """
        Create a copy of the adduct.
        
        Returns:
            Adduct: A new Adduct instance with the same properties.
        """
        adduct = Adduct(
            ion_type=self._ion_type,
            n_molecules=self._n_molecules,
            adducts_in=[f.copy() for f in self._adduct_formulas.keys() if self._adduct_formulas[f] > 0],
            adducts_out=[f.copy() for f in self._adduct_formulas.keys() if self._adduct_formulas[f] < 0],
            charge_offset=0
        )
        adduct._charge = self._charge
        return adduct

    @staticmethod
    def parse(adduct_str: str) -> "Adduct":
        """
        Parse an adduct string like "[M+HCOOH-H]-" or "[2F-H]-" into an Adduct object.

        Args:
            adduct_str (str): The adduct string (e.g., "[M+H]+", "[2M-H]-", "[3F+Na]+").

        Returns:
            Adduct: Parsed Adduct object.
        """
        assert adduct_str.startswith("[") and "]" in adduct_str, f"Invalid adduct format: {adduct_str}"

        # --- Extract core parts ---
        main, charge_part = adduct_str[1:].split("]")
        charge = charge_from_str(charge_part.strip())

        # --- Detect n + type (e.g. 2M, 3F, NP, etc.) ---
        n_match = re.match(r"(\d*)([A-Za-z]+)", main)
        if not n_match:
            raise ValueError(f"Invalid adduct format: missing molecule identifier in {adduct_str}")

        n_molecules = int(n_match.group(1)) if n_match.group(1) else 1
        ion_type = n_match.group(2)
        remainder = main[n_match.end():] 

        # --- Parse adduct formulas ---
        pattern = re.compile(r'([+-])(\d*)([A-Z][A-Za-z0-9]*)|([+-])(\d*)(i)')
        adducts_in, adducts_out = [], []

        for sign, num, formula_str, i_sign, i_num, i_formula_str in pattern.findall(remainder):
            if i_formula_str == 'i':  # neutron case
                formula_str = f"+{Formula.neutron().symbol}"
                sign = i_sign
                num = i_num
            count = int(num) if num else 1
            formulas = [Formula.parse(formula_str) for _ in range(count)]
            if sign == "+":
                adducts_in.extend(formulas)
            else:
                adducts_out.extend(formulas)

        # --- Construct Adduct object ---
        adduct = Adduct(
            ion_type=ion_type,
            n_molecules=n_molecules,
            adducts_in=adducts_in,
            adducts_out=adducts_out,
            charge_offset=0
        )
        adduct._charge = charge

        return adduct

    @property
    def element_diff(self) -> dict[str, int]:
        """
        Return the total difference in element counts caused by the adduct.
        
        Returns:
            dict[str, int]: Dictionary of element symbol to count (can be negative).
        """
        total: dict[str, int] = defaultdict(int)

        for formula, count in self._adduct_formulas.items():
            for elem, elem_count in formula.elements.items():
                total[elem] += elem_count * count

        return dict(total)
    
    def calc_formula(self, neutral_formula: Formula) -> Formula:
        """
        Calculate the resulting formula after adduct formation.

        Args:
            neutral_formula (Formula): Monoisotopic neutral formula of the molecule.

        Returns:
            Formula: Resulting formula after adduct addition.
        """
        total_formula = neutral_formula * self._n_molecules + self.formula_shift
        return total_formula
    
    def calc_mass(self, neutral_mass: float) -> float:
        """
        Calculate the total mass after adduct formation (without dividing by charge).

        Args:
            neutral_mass (float): Monoisotopic neutral mass of the molecule.

        Returns:
            float: Total combined mass after adduct addition.
        """
        total_mass = neutral_mass * self._n_molecules + self.mass_shift
        return total_mass

    def calc_mz(self, neutral_mass: float) -> float:
        """
        Calculate the observed m/z after adduct formation.

        Args:
            neutral_mass (float): Monoisotopic neutral mass of the molecule.

        Returns:
            float: Observed m/z value.
        """
        total_mass = self.calc_mass(neutral_mass)

        if self.charge == 0:
            raise ValueError(f"Cannot calculate m/z for uncharged adduct: {self}")

        mz = total_mass / abs(self.charge)
        return mz