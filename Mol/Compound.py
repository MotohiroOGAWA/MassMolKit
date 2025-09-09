from __future__ import annotations

from typing import Union, Tuple, Dict
from rdkit import Chem
from rdkit.Chem import ResonanceMolSupplier, ResonanceFlags
from .Formula import Formula
from typing import Optional
from collections import defaultdict
from bidict import bidict

class Compound:
    """
    Compound class to represent a chemical structure.
    """

    def __init__(self, mol: Chem.Mol, overwrite_atom_map: bool = False):
        assert isinstance(mol, Chem.Mol), "mol must be a SMILES string or an RDKit Mol object"

        # if the input is an RDKit Mol object
        smiles = Chem.MolToSmiles(mol, canonical=True)
        self._mol = Chem.MolFromSmiles(smiles) 
        
        self.with_atom_map(inplace=True, overwrite=overwrite_atom_map)
        self._formula = Formula.from_mol(self._mol)
        
    def __repr__(self):
        return f"Compound(smiles={self.smiles})"
    
    def __str__(self):
        return self.smiles
    
    @staticmethod
    def from_smiles(smiles: str, overwrite_atom_map: bool = False) -> 'Compound':
        """
        Create a Compound instance from a SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles)
        return Compound(mol, overwrite_atom_map=overwrite_atom_map)
    
    @property
    def mol(self) -> Chem.Mol:
        """
        Get the RDKit Mol object of the compound.
        """
        _mol = Chem.Mol(self._mol)  # Create a copy to avoid modifying the original
        for atom in _mol.GetAtoms():
            atom.SetAtomMapNum(0)  # Reset atom map numbers to 0
        return _mol
    
    @property
    def smiles(self) -> str:
        """
        Get the SMILES representation of the compound.
        """
        return Chem.MolToSmiles(self.mol, canonical=True)
    
    @property
    def _smiles(self) -> str:
        """
        Get the SMILES with atom map of the compound.
        """
        return Chem.MolToSmiles(self._mol, canonical=True)
    
    @property
    def formula(self) -> Formula:
        """
        Get the molecular formula of the compound.
        """
        return self._formula
    
    @property
    def atom_map_to_idx(self) -> bidict[int, int]:
        """
        Get a mapping from atom map numbers to canonical atom indices.
        This maps the atom map numbers assigned before canonicalization
        to the new atom indices after canonical SMILES generation.
        """
        _mol = Chem.Mol(self._mol)  # Create a copy to avoid modifying the original

        # Preserve original atom map numbers
        old_atom_map_num = {atom.GetIdx(): atom.GetAtomMapNum() for atom in _mol.GetAtoms() if atom.GetAtomMapNum() > 0}

        # Remove atom map numbers for canonicalization
        for atom in _mol.GetAtoms():
            atom.SetAtomMapNum(0)

        # Generate canonical SMILES (this triggers atom ordering)
        Chem.MolToSmiles(_mol, canonical=True)

        # Retrieve canonical atom order
        atom_order = list(map(int, _mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))

        # Build mapping: atom_map_num → new atom index
        atom_map_to_idx = {}
        for new_idx, old_idx in enumerate(atom_order):
            atom_map_num = old_atom_map_num.get(old_idx, None)
            if atom_map_num is not None:
                atom_map_to_idx[atom_map_num] = new_idx

        return bidict(atom_map_to_idx)
    
    @property
    def charge(self) -> int:
        """
        Get the charge of the compound.
        """
        return sum(atom.GetFormalCharge() for atom in self._mol.GetAtoms())
    
    @property
    def exact_mass(self) -> float:
        """
        Get the exact mass of the compound.
        """
        return self.formula.exact_mass

    def with_atom_map(self, inplace: bool = False, overwrite: bool = False, atom_map_dict: Optional[Dict[int, int]] = None) -> Optional['Compound']:
        """
        Add atom map numbers to atoms and optionally track old->new mapping.

        Args:
            inplace (bool): Modify compound in place if True.
            overwrite (bool): If True, overwrite all existing map numbers.
            atom_map_dict (dict, optional): Dict to store old->new atom map number mappings.

        Returns:
            Compound or None: Modified compound if not inplace, otherwise None.
        """
        assert (atom_map_dict is None) or (isinstance(atom_map_dict, dict) and len(atom_map_dict) == 0), "atom_map_dict must be a dict or empty if provided"

        # Use a copy or the original mol
        mol = self._mol if inplace else Chem.Mol(self._mol)
        n_atoms = mol.GetNumAtoms()
        
        used_map_nums = set()
        if not overwrite:
            used_map_nums = {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            max_map_num = max(used_map_nums, default=0)
            diff = set(range(1, max_map_num)).difference(used_map_nums)
            next_map_nums = list(diff) + list(range(max_map_num + 1, max_map_num + n_atoms + 1))
        else:
            next_map_nums = list(range(1, n_atoms + 1))
        
        for atom in mol.GetAtoms():
            if overwrite or atom.GetAtomMapNum() == 0:
                old_map_num = atom.GetAtomMapNum()
                new_map_num = next_map_nums.pop(0)
                atom.SetAtomMapNum(new_map_num)
                if atom_map_dict is not None and old_map_num > 0:
                    atom_map_dict[old_map_num] = new_map_num

        if inplace:
            self._mol = mol
        else:
            return Compound(mol)

    def copy(self) -> 'Compound':
        """
        Create a copy of the compound.
        """
        return Compound(self._mol)

    def get_atom_index_from_map(self, map_num: int) -> Optional[int]:
        """
        Get the atom index from a given atom map number.
        """
        for atom in self._mol.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                return atom.GetIdx()
        return None  # Not found
        


    