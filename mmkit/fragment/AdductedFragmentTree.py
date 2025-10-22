from typing import Tuple, Dict, List

from ..mass.Adduct import Adduct
from ..mass.constants import AdductType, IonMode
from .FragmentTree import *
from ..chem.Compound import Compound
from ..chem.Formula import Formula


class AdductedFragmentTree:
    """
    Represents a FragmentTree annotated with adduct information.
    This class enables theoretical m/z estimation for each fragment node
    based on its molecular formula and charge state.
    """
    SUPPORTED_ADDUCT_TYPES_POS = [AdductType.M_PLUS_H_POS, AdductType.M_PLUS_NH4_POS, AdductType.M_PLUS_Na_POS]
    SUPPORTED_ADDUCTS_POS = {adduct_type: Adduct.parse(adduct_type.value) for adduct_type in SUPPORTED_ADDUCT_TYPES_POS}
    EMPTY_ADDUCT_POS1 = Adduct.parse('[M]+')
    EMPTY_ADDUCT_NEG1 = Adduct.parse('[M]-')

    def __init__(self, fragment_tree: FragmentTree):
        """
        Initialize the AdductedFragmentTree.
        
        Args:
            fragment_tree (FragmentTree): The base fragment tree to annotate.
        """
        self.fragment_tree = fragment_tree
        # Maps (formula, charge) → list of node indices that share this pair
        self._formula_charge_index_map: Dict[Tuple[Formula, int], List[int]] = None
    
    @property
    def nodes(self) -> Dict[int, FragmentNode]:
        return self.fragment_tree.nodes

    @property
    def edges(self) -> Dict[Tuple[int, int], FragmentEdge]:
        return self.fragment_tree.edges

    # -------------------------------------------------------------------------
    def _build_formula_charge_index_map(self) -> None:
        """
        Build a lookup table mapping (Formula, charge) pairs
        to the list of fragment node indices having that combination.
        
        This enables quick access to all fragments with the same composition
        when computing m/z values or grouping equivalent fragments.
        """
        self._formula_charge_index_map = {}

        for idx, node in self.fragment_tree.nodes.items():
            compound = Compound.from_smiles(node.smiles)
            formula = compound.formula
            charge = compound.charge
            key = (formula, charge)

            if key not in self._formula_charge_index_map:
                self._formula_charge_index_map[key] = []

            self._formula_charge_index_map[key].append(idx)

    def get_all_formulas_with_node_id(self, adduct_type:AdductType) -> Dict[str, Tuple[Formula, List[int]]]:
        if adduct_type in self.SUPPORTED_ADDUCT_TYPES_POS:
            ion_mode = IonMode.POSITIVE
            charge_mode = 1
            adducts = [self.SUPPORTED_ADDUCTS_POS[at] for at in set([adduct_type, AdductType.M_PLUS_H_POS])]
            empty_adduct = self.EMPTY_ADDUCT_POS1
        else:
            raise ValueError(f"Adduct type {adduct_type} is not supported.")
        
        if self._formula_charge_index_map is None:
            self._build_formula_charge_index_map()
        
        all_formulas = {}
        for (formula, charge), node_indices in self._formula_charge_index_map.items():
            adducted_formulas = []
            if charge == 0:
                for adduct in adducts:
                    adducted_formula = adduct.calc_formula(formula)
                    adducted_formulas.append(adducted_formula)
            elif charge == charge_mode:
                adducted_formula = empty_adduct.calc_formula(formula)
                adducted_formulas.append(adducted_formula)
            else:
                raise ValueError(f"Unsupported charge state {charge} for adduct type {adduct_type}.")
            
            for adducted_formula in adducted_formulas:
                if adducted_formula not in all_formulas:
                    all_formulas[adducted_formula] = []
                all_formulas[adducted_formula].extend(node_indices)

        sorted_formulas = {str(f): (f, resource) for f, resource in dict(sorted(all_formulas.items(), key=lambda kv: kv[0].exact_mass)).items()}
        return sorted_formulas  
    
    def get_all_formulas(self, adduct_type:AdductType) -> Tuple[Formula]:
        formula_with_node_id = self.get_all_formulas_with_node_id(adduct_type)
        return (v[0] for v in formula_with_node_id.values())




            
        
    