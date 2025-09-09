import re
from collections import OrderedDict
from typing import Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

class Formula:
    def __init__(self, elements: Dict[str, int], charge: int, raw_formula: str = ""):
        # OrderedDict to preserve Hill order: C, H, then alphabetical
        self.elements: OrderedDict[str, int]
        self.charge: int = charge
        self.raw_formula: str = raw_formula

        self._reorder_elements(elements.copy())
        

    @property
    def exact_mass(self) -> float:
        """
        Calculate the exact mass of the formula.
        """
        mass = 0.0
        for elem, count in self.elements.items():
            atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(elem)
            mass += Chem.GetPeriodicTable().GetMostCommonIsotopeMass(atomic_number) * count
        return mass

    def __repr__(self):
        return f"Formula({self.__str__()})"
    

    def __str__(self) -> str:
        return self.to_string(no_charge=False)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.elements.items()), self.raw_formula, self.charge))

    def _parse_formula(self, formula: str):
        """
        Parse chemical formula and set element order according to Hill system.
        """
        # Extract and remove charge
        charge_match = re.search(r"([+-]+|[+-]\d+)$", formula)
        if charge_match:
            charge_str = charge_match.group(1)
            formula = formula[: -len(charge_str)]
            self.charge = int(charge_str[1:]) if charge_str[1:] else 1
            if charge_str[0] == '-':
                self.charge *= -1

        self.raw_formula = formula  # Store the raw formula for reference

        # Parse element counts
        matches = re.findall(r"([+-]?)([A-Z][a-z]?)(\d*)", formula)
        temp = {}
        for sign, elem, count in matches:
            count = int(count) if count else 1
            if sign == "-":
                count = -count
            temp[elem] = temp.get(elem, 0) + count
        temp = {k: v for k, v in temp.items() if v != 0}

        # Determine Hill order
        keys = temp.keys()
        ordered = Formula._reorder_element_keys(keys)

        # Store in ordered dict
        self.elements = OrderedDict((k, temp[k]) for k in ordered)

    def __add__(self, other: 'Formula') -> 'Formula':
        result = Formula(elements={}, charge=0)
        combined = dict(self.elements)

        for elem, count in other.elements.items():
            combined[elem] = combined.get(elem, 0) + count

        result.charge = self.charge + other.charge
        result._reorder_elements(combined)
        return result

    def __sub__(self, other: 'Formula') -> 'Formula':
        result = Formula(elements={}, charge=0)
        combined = dict(self.elements)

        for elem, count in other.elements.items():
            combined[elem] = combined.get(elem, 0) - count

        result.charge = self.charge - other.charge
        result._reorder_elements(combined)
        return result
    
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

        self.elements = OrderedDict((k, element_counts[k]) for k in ordered if element_counts[k] != 0)

    @staticmethod
    def _reorder_element_keys(elements: list[str]) -> OrderedDict:
        """
        Reorder elements according to Hill system.
        """
        mol = Chem.RWMol()
        for elem in elements:
            atom = Chem.Atom(elem)
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
        mol = mol.GetMol()

        formula_str = rdMolDescriptors.CalcMolFormula(mol)
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula_str)

        ordered = tuple(m[0] for m in matches)
        return ordered

    def to_string(self, no_charge: bool = False) -> str:
        parts = []
        for elem, count in self.elements.items():
            if count > 0:
                parts.append(f"{elem}{count if count != 1 else ''}")
            elif count < 0:
                parts.append(f"-{elem}{-count if count != -1 else ''}")
        formula = "".join(parts)
        
        if not no_charge:
            if self.charge > 0:
                formula += ("+" if self.charge == 1 else f"+{self.charge}")
            elif self.charge < 0:
                formula += ("-" if self.charge == -1 else f"-{-self.charge}")
        return formula

    def copy(self) -> 'Formula':
        return Formula(self.elements, self.charge, self.raw_formula)
    
    @classmethod
    def from_str(self, formula_str: str) -> 'Formula':
        """
        Create a Formula object from a formula string.
        """
        f = Formula(elements={}, charge=0)
        f._parse_formula(formula_str)
        return f
    
    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> 'Formula':
        """
        Create a Formula object from an RDKit Mol object.
        """
        formula_str = rdMolDescriptors.CalcMolFormula(mol)
        return cls(formula_str)

    