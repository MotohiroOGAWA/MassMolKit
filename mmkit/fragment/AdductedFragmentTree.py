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

    @property
    def compound(self) -> Compound:
        return self.fragment_tree.compound
    
    @property
    def nodes(self) -> Dict[int, FragmentNode]:
        return self.fragment_tree.nodes

    @property
    def edges(self) -> Dict[Tuple[int, int], FragmentEdge]:
        return self.fragment_tree.edges

    def get_all_formulas_with_node_id(self, precursor_type: Adduct) -> Dict[str, Tuple[Formula, Adduct, List[int]]]:
        adduct_composition, neutral_component_adduct = precursor_type.split_adduct_components()
        assert len(adduct_composition) == 1, f"Only one neutral component is supported: {precursor_type}, {adduct_composition}"
        adduct_type = next(iter(adduct_composition))
        assert adduct_composition[adduct_type] == 1, f"Only single charge adducts are supported: {precursor_type}, {adduct_composition}"

        if adduct_type in self.SUPPORTED_ADDUCT_TYPES_POS:
            ion_mode = IonMode.POSITIVE
            charge_mode = 1
            adducts = [self.SUPPORTED_ADDUCTS_POS[at] for at in set([adduct_type, AdductType.M_PLUS_H_POS])]
            empty_adduct = self.EMPTY_ADDUCT_POS1
        else:
            raise ValueError(f"Adduct type {adduct_type} is not supported.")
        
        element_neutron_count = neutral_component_adduct.element_diff.get('n', 0)
        if element_neutron_count < 0 or element_neutron_count % 2 != 0:
            raise ValueError(f"Unsupported neutral component adduct with negative or odd neutron count: {neutral_component_adduct}")
        root_isotope_count = element_neutron_count // 2
        def get_formula_halogen_count(formula:Formula) -> int:
            count = 0
            if 'Cl' in formula:
                count += formula.elements.get('Cl')
            if 'Br' in formula:
                count += formula.elements.get('Br')
            return count
        root_n_halogens = get_formula_halogen_count(self.compound.formula)

        all_formulas = {}
        if ion_mode == IonMode.POSITIVE:
            for formula, node_indices in self.fragment_tree.formula_index_map.items():
                adduct_formula_pairs = []
                n_halogens = get_formula_halogen_count(formula)
                min_isotope_count = max(0, root_isotope_count - (root_n_halogens - n_halogens))
                max_isotope_count = min(root_isotope_count, n_halogens)
                for iso_count in range(min_isotope_count, max_isotope_count + 1):
                    f = formula.copy()
                    if iso_count > 0:
                        f = f + (Formula.parse('+n')*iso_count * 2)
                    if f.charge == 0:
                        for adduct in adducts:
                            adduct_formula_pairs.append((f, adduct))
                    elif f.charge == charge_mode:
                        adduct_formula_pairs.append((f, empty_adduct))
                    else:
                        raise ValueError(f"Unsupported charge state {formula.charge} for adduct type {adduct_type}.")
                
                for adduct_formula_pair in adduct_formula_pairs:
                    adducted_formula = adduct_formula_pair[1].calc_formula(adduct_formula_pair[0])
                    if adducted_formula not in all_formulas:
                        all_formulas[adducted_formula] = {}
                    if adduct_formula_pair not in all_formulas[adducted_formula]:
                        all_formulas[adducted_formula][adduct_formula_pair] = []
                    all_formulas[adducted_formula][adduct_formula_pair].extend(node_indices)

        sorted_formulas = {k: v for k, v in dict(sorted(all_formulas.items(), key=lambda kv: kv[0].exact_mass)).items()}
        return sorted_formulas

    def get_all_formulas(self, precursor_type: Adduct) -> Tuple[Formula]:
        formula_with_node_id = self.get_all_formulas_with_node_id(precursor_type)
        return (k for k in formula_with_node_id.keys())




            
        
    