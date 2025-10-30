import re
from collections import OrderedDict
from typing import Dict, Union
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class Neutron:
    """Virtual representation of a neutron particle."""
    symbol = "n"
    # mass = 1.008665  # Atomic mass in daltons (Da)
    mass = 0.9987508525  # Halogen  mass in daltons (Da)
    charge = 0       # Neutral charge
    atomic_num = 0   # Not an element in the periodic table
neutron = Neutron()

class Formula:
    def __init__(self, elements: Dict[str, int], charge: int, raw_formula: str = ""):
        # OrderedDict to preserve Hill order: C, H, then alphabetical
        self._elements: OrderedDict[str, int]
        self._charge: int = charge
        self._raw_formula: str = raw_formula

        self._reorder_elements(elements.copy())

    @property
    def elements(self) -> Dict[str, int]:
        """
        Return a dictionary of elements and their counts.
        """
        return OrderedDict(self._elements)

    @property
    def charge(self) -> int:
        """
        Return the net charge of the formula.
        """
        return self._charge
    
    @property
    def raw_formula(self) -> str:
        """
        Return the raw formula string as provided during initialization.
        """
        return self._raw_formula

    @property
    def exact_mass(self) -> float:
        """
        Calculate the exact mass of the formula.
        """
        mass = 0.0
        for elem, count in self._elements.items():
            if elem == neutron.symbol:  # neutron
                mass += neutron.mass * count
            else:
                atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(elem)
                mass += Chem.GetPeriodicTable().GetMostCommonIsotopeMass(atomic_number) * count
        return mass

    @property
    def is_nonnegative(self) -> bool:
        """
        Return True if all element counts are non-negative (>= 0).

        Example:
            >>> Formula.parse("C6H12O6").is_nonnegative
            True
            >>> Formula.parse("C-1H2O").is_nonnegative
            False
        """
        return all(count >= 0 for count in self._elements.values())

    def __repr__(self):
        return f"Formula({self.__str__()})"
    

    def __str__(self) -> str:
        return self.to_string(no_charge=False)
    
    def __hash__(self) -> int:
        return hash((frozenset(self._elements.items()), self._raw_formula, self._charge))
    
    def __contains__(self, item: Union['Formula', str]) -> bool:
        if not self.is_nonnegative:
            raise ValueError("Containment check is only supported for non-negative formulas.")
        
        if isinstance(item, str):
            item = Formula.parse(item)
        elif not isinstance(item, Formula):
            raise TypeError(f"Containment check only supports Formula or str, not {type(item)}")
        
        return all(self._elements.get(elem, 0) >= count for elem, count in item._elements.items())

    def _parse_formula(self, formula: str):
        """
        Parse chemical formula and set element order according to Hill system.
        """
        # Extract and remove charge
        charge_match = re.search(r"([+-]+|[+-]\d+)$", formula)
        if charge_match:
            charge_str = charge_match.group(1)
            formula = formula[: -len(charge_str)]
            self._charge = int(charge_str[1:]) if charge_str[1:] else 1
            if charge_str[0] == '-':
                self._charge *= -1

        self._raw_formula = formula  # Store the raw formula for reference

        # Parse element counts
        matches = re.findall(rf"([+-]?)([A-Z][a-z]?)(\d*)|([+-])({neutron.symbol})(\d*)", formula)
        temp = {}
        for sign, elem, count, i_sign, i_elem, i_count in matches:
            if i_elem == neutron.symbol:  # neutron
                elem = neutron.symbol
                sign = i_sign
                count = i_count
            count = int(count) if count else 1
            if sign == "-":
                count = -count
            temp[elem] = temp.get(elem, 0) + count
        temp = {k: v for k, v in temp.items() if v != 0}

        # Determine Hill order
        keys = temp.keys()
        ordered = Formula._reorder_element_keys(keys)

        # Store in ordered dict
        self._elements = OrderedDict((k, temp[k]) for k in ordered)

    def __add__(self, other: 'Formula') -> 'Formula':
        result = Formula(elements={}, charge=0)
        combined = dict(self._elements)

        for elem, count in other._elements.items():
            combined[elem] = combined.get(elem, 0) + count

        result._charge = self._charge + other._charge
        result._reorder_elements(combined)
        return result

    def __sub__(self, other: 'Formula') -> 'Formula':
        result = Formula(elements={}, charge=0)
        combined = dict(self._elements)

        for elem, count in other._elements.items():
            combined[elem] = combined.get(elem, 0) - count

        result._charge = self._charge - other._charge
        result._reorder_elements(combined)
        return result
    
    def __mul__(self, factor: int) -> "Formula":
        """
        Multiply formula by an integer factor.
        Example: H2O * 2 -> H4O2
        """
        if not isinstance(factor, int):
            raise TypeError(f"Formula can only be multiplied by int, not {type(factor)}")

        new_elements = {elem: count * factor for elem, count in self._elements.items()}
        new_charge = self._charge * factor
        return Formula(new_elements, new_charge, self._raw_formula)

    def __rmul__(self, factor: int) -> "Formula":
        """Support reversed multiplication: 2 * Formula(...)"""
        return self.__mul__(factor)
    
    def __eq__(self, other: 'Formula') -> bool:
        if not isinstance(other, Formula):
            return False
        
        return str(self) == str(other)
    
    @property
    def value(self) -> str:
        """
        Return the formula as a string with charge.
        """
        return self.to_string(no_charge=False)
    
    @property
    def plain(self) -> str:
        """
        Return the formula as a plain string without charge.
        """
        return self.to_string(no_charge=True)

    def _reorder_elements(self, element_counts: Dict[str, int]):
        """Apply Hill system ordering to elements and store as OrderedDict."""
        keys = element_counts.keys()
        ordered = Formula._reorder_element_keys(keys)

        self._elements = OrderedDict((k, element_counts[k]) for k in ordered if element_counts[k] != 0)

    @staticmethod
    def _reorder_element_keys(elements: list[str]) -> OrderedDict:
        """
        Reorder elements according to Hill system.
        """
        mol = Chem.RWMol()
        exist_neutron = False
        for elem in elements:
            if elem == neutron.symbol:  # neutron
                exist_neutron = True
                continue
            atom = Chem.Atom(elem)
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
        mol = mol.GetMol()

        formula_str = rdMolDescriptors.CalcMolFormula(mol)
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)

        ordered = tuple(m[0] for m in matches)
        if exist_neutron:
            ordered += (neutron.symbol,)
        return ordered

    def to_string(self, no_charge: bool = False) -> str:
        parts = []
        neutron_formula = ''
        for elem, count in self._elements.items():
            if elem == neutron.symbol:  # neutron
                if count > 0:
                    neutron_formula = f"+{elem}{count if count != 1 else ''}"
                elif count < 0:
                    neutron_formula = f"-{elem}{-count if count != -1 else ''}"
            else:
                if count > 0:
                    parts.append(f"{elem}{count if count != 1 else ''}")
                elif count < 0:
                    parts.append(f"-{elem}{-count if count != -1 else ''}")
        formula = "".join(parts)
        formula += neutron_formula
        
        if not no_charge:
            if self._charge > 0:
                formula += ("+" if self._charge == 1 else f"+{self._charge}")
            elif self._charge < 0:
                formula += ("-" if self._charge == -1 else f"-{-self._charge}")
        return formula

    def copy(self) -> 'Formula':
        return Formula(self._elements, self._charge, self._raw_formula)
    
    @classmethod
    def parse(self, formula_str: str, store_raw: bool = True) -> 'Formula':
        """
        Create a Formula object from a formula string.

        Args:
            formula_str (str): Chemical formula string to parse.
            store_raw (bool): If True, save the original input string to _raw_formula.
                            If False, _raw_formula will remain an empty string.

        Returns:
            Formula: Parsed Formula object.

        Example:
            >>> Formula.parse("C6H12O6").raw_formula
            'C6H12O6'
            >>> Formula.parse("C6H12O6", store_raw=False).raw_formula
            ''
        """
        f = Formula(elements={}, charge=0)
        f._parse_formula(formula_str)
        if store_raw:
            f._raw_formula = formula_str
        else:
            f._raw_formula = ""
        return f
    
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> 'Formula':
        """
        Create a Formula object from an RDKit Mol object.
        Includes implicit hydrogens by adding them explicitly.
        """
        # Make a copy and add explicit hydrogens
        mol_with_H = Chem.AddHs(mol)

        elements = {}
        total_charge = 0

        for atom in mol_with_H.GetAtoms():
            symbol = atom.GetSymbol()
            elements[symbol] = elements.get(symbol, 0) + 1
            total_charge += atom.GetFormalCharge()

        return cls(elements=elements, charge=total_charge)
    
    @staticmethod
    def empty() -> 'Formula':
        """
        Return an empty Formula object (no elements, zero charge).
        """
        return Formula(elements={}, charge=0)
    
    @staticmethod
    def neutron() -> 'Neutron':
        """
        Return a virtual Neutron object.
        """
        return Neutron()