from __future__ import annotations
from typing import Union, List, Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import ResonanceMolSupplier, ResonanceFlags
from collections import Counter, deque, defaultdict
from bidict import bidict

from ..Mol.Compound import Compound
from ..MS.constants import AdductType
from ..MS.Adduct import Adduct
from ..MS.AdductIon import AdductIon

# Suppress warnings and informational messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

class Frag:
    def __init__(self, fragment: Compound, adduct_type: AdductType):
        
        self._raw: Compound = fragment # original fragment with dummy atoms
        self._adduct_type = adduct_type
        self._candidates: Dict[Compound, List[Adduct]] = {}

        self._reconstruct_fragment()

    def __repr__(self):
        adduct_ions = [AdductIon(c, a) for c, adducts in self._candidates.items() for a in adducts]
        return f"Frag({tuple(adduct_ions)})"

    def _reconstruct_fragment(self):
        """
        Reconstruct a fragmented molecule to satisfy valency and charge stability.
        """
        rw_mol = Chem.RWMol(self._raw._mol)

        dummy_ids = [atom.GetAtomMapNum() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        all_symbols = [atom.GetSymbol() for atom in rw_mol.GetAtoms()]

        adduct = None
        next_m = None
        candidates = defaultdict(list)
        for dummy_id in dummy_ids:
            if next_m is None:
                _rw_mol = Chem.RWMol(rw_mol)  # Create a copy of the molecule to avoid modifying the original
            else:
                _rw_mol = Chem.RWMol(next_m)
            
            mapper = bidict({atom.GetAtomMapNum(): atom.GetIdx() for atom in _rw_mol.GetAtoms() if atom.GetAtomMapNum() != 0})
            dummy_atom = _rw_mol.GetAtomWithIdx(mapper[dummy_id])
            neighbors = dummy_atom.GetNeighbors()
            if len(neighbors) != 1:
                raise ValueError("Expected exactly one neighbor for the dummy atom")

            connected_atom = neighbors[0]

            symbol = connected_atom.GetSymbol()
            charge = connected_atom.GetFormalCharge()
            total_valence = connected_atom.GetTotalValence()
            fragmented_valence = total_valence - 1  # Assume one bond is missing

            # is_stable = Atom.is_stable(symbol, charge, fragmented_valence)
            total_charge = sum(atom.GetFormalCharge() for atom in _rw_mol.GetAtoms())

            if self._adduct_type == AdductType.M_PLUS_H_POS:
                if (total_charge == 0) and (symbol == "C"):
                    next_m = self._replace_dummy_with_hydrogen(_rw_mol, dummy_id)
                    adduct = Adduct.parse("[F-H]+")

                elif any(sym in ["C", "O", "N"] for sym in all_symbols):
                    next_m = self._replace_dummy_with_hydrogen(_rw_mol, dummy_id)
                    if adduct is None:
                        adduct = Adduct.parse("[F+H]+")
                else:
                    raise NotImplementedError(f"Reconstruction not implemented for atom type: {symbol}")
            else:
                raise NotImplementedError(f"Reconstruction not implemented for adduct type: {self._adduct_type}")

        # Not Fragmented or no valid reconstruction found
        if next_m is None:
            c = self._raw.copy()
            if c not in self._candidates:
                candidates[c] = []
            if self._adduct_type == AdductType.M_PLUS_H_POS:
                candidates[c].append(Adduct.parse("[M+H]+"))
            else:
                raise NotImplementedError(
                    f"reconstruct_single_bond_fragment: Unsupported adduct type '{self._adduct_type}' for stable molecule with no charge."
                )
        else:
            c = Compound(next_m)
            if adduct not in candidates[c]:
                candidates[c].append(adduct)

        self._candidates = dict(candidates)


    def ___reconstruct_fragment(self):
        """
        Reconstruct a fragmented molecule to satisfy valency and charge stability.
        """
        rw_mol = Chem.RWMol(self._raw._mol)

        dummy_ids = [atom.GetAtomMapNum() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 0]
        all_symbols = [atom.GetSymbol() for atom in rw_mol.GetAtoms()]

        scores = []
        next_m = None
        candidates = defaultdict(list)
        for i in range(2) if len(dummy_ids) > 1 else range(1, 2):
            for dummy_id in dummy_ids:
                if next_m is None:
                    _rw_mol = Chem.RWMol(rw_mol)  # Create a copy of the molecule to avoid modifying the original
                else:
                    _rw_mol = Chem.RWMol(next_m)
                
                mapper = bidict({atom.GetAtomMapNum(): atom.GetIdx() for atom in _rw_mol.GetAtoms() if atom.GetAtomMapNum() != 0})
                dummy_atom = _rw_mol.GetAtomWithIdx(mapper[dummy_id])
                neighbors = dummy_atom.GetNeighbors()
                if len(neighbors) != 1:
                    raise ValueError("Expected exactly one neighbor for the dummy atom")

                connected_atom = neighbors[0]

                symbol = connected_atom.GetSymbol()
                charge = connected_atom.GetFormalCharge()
                total_valence = connected_atom.GetTotalValence()
                fragmented_valence = total_valence - 1  # Assume one bond is missing

                # is_stable = Atom.is_stable(symbol, charge, fragmented_valence)
                total_charge = sum(atom.GetFormalCharge() for atom in _rw_mol.GetAtoms())

                if self._adduct_type == AdductType.M_PLUS_H_POS:
                    if (total_charge == 0) and (symbol == "C"):
                        _rw_mol.RemoveAtom(dummy_atom.GetIdx())
                        connected_atom.SetFormalCharge(+1)
                        routes = self._find_all_charge_shift_path(_rw_mol, connected_atom.GetAtomMapNum())
                        scores.extend([(dummy_id, route) for route in routes])
                        if i == 1:
                            _rw_mol = self._shift_charge_along_path(_rw_mol, routes[0]["path"])
                            next_m = _rw_mol.GetMol()

                    elif i == 1 and any(sym in ["C", "O", "N"] for sym in all_symbols):
                        next_m = self._replace_dummy_with_hydrogen(_rw_mol, dummy_id)
                        
                    elif i == 0:
                        pass
                    else:
                        raise NotImplementedError(f"Reconstruction not implemented for atom type: {symbol}")
                else:
                    raise NotImplementedError(f"Reconstruction not implemented for adduct type: {self._adduct_type}")
                
            if i == 0 and len(scores) > 0:
                # Sort by priority and take the best route
                scores = sorted(scores, key=lambda x: self._calc_charge_shift_priority(x[1]))
                best_dummy_id, best_route = scores[0]
                other_dummy_ids = [id for id in dummy_ids if id != best_dummy_id]
                dummy_ids = [best_dummy_id] + other_dummy_ids

        # Not Fragmented or no valid reconstruction found
        if next_m is None:
            c = self._raw.copy()
            if c not in self._candidates:
                self._candidates[c] = []
            if self._adduct_type == AdductType.M_PLUS_H_POS:
                self._candidates[c].append(Adduct.parse("[M+H]+"))
            else:
                raise NotImplementedError(
                    f"reconstruct_single_bond_fragment: Unsupported adduct type '{self._adduct_type}' for stable molecule with no charge."
                )
        else:
            c = Compound(next_m)
            if c.charge == 0:
                a = Adduct.parse("[F+H]+")
            elif c.charge == 1:
                a = Adduct.parse("[F]+")
            else:
                raise NotImplementedError(f"Reconstruction not implemented for molecule with charge {c.charge}")
            
            if a not in candidates[c]:
                candidates[c].append(a)

        self._candidates = dict(candidates)

    def _replace_dummy_with_hydrogen(self, rw_mol: Chem.RWMol, dummy_id: int) -> Chem.Mol:
        """
        Replace a dummy atom in the molecule with a hydrogen atom.

        Args:
            rw_mol (Chem.RWMol): The mutable molecule object.
            dummy_id (int): The index of the dummy atom to be replaced.

        Returns:
            Chem.RWMol: The modified molecule with dummy replaced by hydrogen.
        """
        mapper = bidict({atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomMapNum() != 0})
        dummy_atom = rw_mol.GetAtomWithIdx(mapper[dummy_id])
        neighbors = dummy_atom.GetNeighbors()
        
        if len(neighbors) != 1:
            raise ValueError(f"Dummy atom at index {dummy_id} should have exactly one neighbor, found {len(neighbors)}")
        
        connected_atom = neighbors[0]

        # Remove dummy atom
        rw_mol.RemoveAtom(mapper[dummy_id])

        # Get the index of the neighbor atom
        connected_atom_idx = connected_atom.GetIdx()

        # Add hydrogen atom and bond it to the former neighbor
        h_idx = rw_mol.AddAtom(Chem.Atom(1))  # AtomicNum 1 = Hydrogen
        rw_mol.AddBond(connected_atom_idx, h_idx, Chem.BondType.SINGLE)
        m = Chem.RemoveHs(rw_mol)
        return m


    def _find_all_charge_shift_path(self, rw_mol: Chem.RWMol, start_id: int) -> List[int]:
        mapper = bidict({atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomMapNum() != 0})
        
        start_atom = rw_mol.GetAtomWithIdx(mapper[start_id])
        if start_atom.GetFormalCharge() == 0:
            return 
        
        if start_atom.GetSymbol() == "O":
            return
        if start_atom.GetSymbol() == "N":
            return
        
        queue = deque([(start_id, [start_id])])  # (current index, path)
        completed_paths = [{
            "path": [start_id],
            "symbol": start_atom.GetSymbol(),
            "carbon_degree": sum(1 for n in start_atom.GetNeighbors() if n.GetSymbol() == "C")
        }]
        while queue:
            current_id, path = queue.popleft()

            current_atom = rw_mol.GetAtomWithIdx(mapper[current_id])
            
            for neighbor in current_atom.GetNeighbors():
                neighbor_id = neighbor.GetAtomMapNum()
                if neighbor_id in path:
                    continue
                
                bond = rw_mol.GetBondBetweenAtoms(mapper[current_id], mapper[neighbor_id])
                if neighbor.GetSymbol() in ["O", "N"] \
                    and bond.GetBondType() in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]:
                    carbon_degree = sum(1 for n in neighbor.GetNeighbors() if n.GetSymbol() == "C")
                    completed_paths.append({
                        "path": path + [neighbor_id],
                        "symbol": neighbor.GetSymbol(),
                        "carbon_degree": carbon_degree
                    })
                    continue
                
                elif (bond.GetBondType() == Chem.BondType.SINGLE)\
                    and neighbor.GetSymbol() == "C"\
                          and neighbor.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                    # If the neighbor is a carbon with SP2 hybridization, continue the search
                    for next_neighbor in neighbor.GetNeighbors():
                        next_id = next_neighbor.GetAtomMapNum()
                        if next_id == current_id:
                            continue

                        next_bond = rw_mol.GetBondBetweenAtoms(mapper[neighbor_id], mapper[next_id])
                        if (next_bond.GetBondType() == Chem.BondType.DOUBLE)\
                              and next_neighbor.GetSymbol() == "C":
                            queue.append((next_id, path + [neighbor_id, next_id]))
                            completed_paths.append({
                                "path": path + [neighbor_id, next_id],
                                "symbol": next_neighbor.GetSymbol(),
                                "carbon_degree": sum(1 for n in next_neighbor.GetNeighbors() if n.GetSymbol() == "C")
                            })

        # Apply sorting based on the defined priority
        completed_paths = sorted(completed_paths, key=self._calc_charge_shift_priority)
        return completed_paths

    def _calc_charge_shift_priority(self, entry: dict) -> tuple:
        """
        Calculate the priority of a charge shift path based on its properties.
        1. Prioritize symbols "O" and "N" over "C"
        2. Prefer higher carbon degree
        3. Prefer shorter paths
        
        Args:
            entry (dict): A dictionary containing the path, symbol, and carbon degree.
        
        Returns:
            tuple: A tuple representing the priority of the entry.
        """
        # Assign lower priority value to "O" and "N", higher to "C", and highest to others
        symbol_priority = 0 if entry["symbol"] in ["O", "N"] else (1 if entry["symbol"] == "C" else 2)
        return (
            symbol_priority,               # Lower is better (O/N preferred)
            -entry["carbon_degree"],       # Higher carbon degree is better
            len(entry["path"])             # Shorter path is better
        )


    def _shift_charge_along_path(self, rw_mol: Chem.RWMol, path: List[int]):
        if len(path) < 2:
            return rw_mol # Cannot shift with a single atom
        
        mapper = bidict({atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomMapNum() != 0})

        current_i = 0
        while current_i < len(path) - 1:
            src_id = path[current_i]
            dst_id = path[current_i + 1]

            src_idx = mapper[src_id]
            dst_idx = mapper[dst_id]
            bond = rw_mol.GetBondBetweenAtoms(src_idx, dst_idx)
            src_atom = rw_mol.GetAtomWithIdx(src_idx)
            dst_atom = rw_mol.GetAtomWithIdx(dst_idx)

            if src_atom.GetSymbol() == 'C' and src_atom.GetFormalCharge() == 1:
                if dst_atom.GetSymbol() == 'C' \
                    and dst_atom.GetFormalCharge() == 0 \
                        and bond.GetBondType() in [Chem.BondType.SINGLE]:
                    
                    nbr_id = path[current_i + 2]
                    nbr_idx = mapper[nbr_id]
                    nbr_atom = rw_mol.GetAtomWithIdx(nbr_idx)
                    nbr_bond = rw_mol.GetBondBetweenAtoms(dst_idx, nbr_idx)
                    if nbr_atom.GetSymbol() == 'C' \
                        and nbr_atom.GetFormalCharge() == 0 \
                            and nbr_bond.GetBondType() in [Chem.BondType.DOUBLE]:
                        # C+–C=C → C=C–C+
                        rw_mol.RemoveBond(src_idx, dst_idx)
                        rw_mol.RemoveBond(dst_idx, nbr_idx)
                        src_atom.SetFormalCharge(0)
                        nbr_atom.SetFormalCharge(1)
                        rw_mol.AddBond(dst_idx, nbr_idx, Chem.BondType.SINGLE)
                        rw_mol.AddBond(src_idx, dst_idx, Chem.BondType.DOUBLE)
                        
                        current_i += 2
                    else:
                        raise NotImplementedError(
                            f"Charge shift not implemented for bond type {bond.GetBondType()} between {src_atom.GetSymbol()} and {dst_atom.GetSymbol()}"
                        )
                elif dst_atom.GetSymbol() in ['O', 'N'] \
                    and dst_atom.GetFormalCharge() == 0:
                    # C+–O → C=O+
                    if bond.GetBondType() in [Chem.BondType.SINGLE]:
                        _bond_type = Chem.BondType.DOUBLE
                    elif bond.GetBondType() == Chem.BondType.DOUBLE:
                        _bond_type = Chem.BondType.TRIPLE
                    else:
                        raise NotImplementedError(
                            f"Charge shift not implemented for bond type {bond.GetBondType()} between {src_atom.GetSymbol()} and {dst_atom.GetSymbol()}"
                        )
                
                    rw_mol.RemoveBond(src_idx, dst_idx)
                    src_atom.SetFormalCharge(0)
                    dst_atom.SetFormalCharge(1)
                    rw_mol.AddBond(src_idx, dst_idx, _bond_type)
                    current_i += 1

                else:
                    raise NotImplementedError(
                        f"Charge shift not implemented for bond type {bond.GetBondType()} between {src_atom.GetSymbol()} and {dst_atom.GetSymbol()}"
                    )
            else:
                raise NotImplementedError(
                    f"Charge shift not implemented for atom {src_atom.GetSymbol()} with charge {src_atom.GetFormalCharge()}"
                )
        Chem.SanitizeMol(rw_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        return rw_mol