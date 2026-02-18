import os
from typing import List, Tuple, Dict, Set, Union
from collections import defaultdict
import yaml
import itertools

from .CleavagePattern import CleavagePattern
from .CleavagePatternSet import CleavagePatternSet
from .HydrogenRearrangement import HydrogenRearrangement
from .FragmentTree import FragmentTree, FragmentNode, FragmentEdge
from .HydrogenRearrangedFragmentTree import HydrogenRearrangedFragmentTree
from .FragmentTreeBuilder import FragmentTreeBuilder
from ..mass.constants import *
from ..chem.Compound import Compound, Formula
from ..chem.formula_utils import assign_formulas_to_peaks
from .FragmentResult import FragmentResult
from .FragmentPathway import *
from ..mass.Adduct import Adduct
from ..mass.utilities import split_adduct_components
from ..mass.Tolerance import MassTolerance

class Fragmenter:
    SUPPORTED_ADDUCT_TYPES: Tuple[Adduct, ...] = (
        Adduct.parse("[M+H]+"),
        Adduct.parse("[M+NH4]+"),
        Adduct.parse("[M+Na]+"),
    )
    def __init__(
            self,
            ion_mode: IonMode,
            fragment_tree_builder: FragmentTreeBuilder,
            adduct_types: Tuple[Adduct],
            hydrogen_rearrangement: HydrogenRearrangement,
            ):
        
        if not isinstance(ion_mode, IonMode):
            ion_mode = parse_ion_mode(str(ion_mode))

        assert isinstance(fragment_tree_builder, FragmentTreeBuilder), "fragment_tree_builder must be an instance of FragmentTreeBuilder"

        # ---- adduct validation ----
        adduct_types = tuple(ad if isinstance(ad, Adduct) else Adduct.parse(ad) for ad in adduct_types)
        unsupported = [
            adduct for adduct in adduct_types
            if adduct not in self.SUPPORTED_ADDUCT_TYPES
        ]
        if unsupported:
            raise ValueError(
                f"Unsupported adduct types: {unsupported}. "
                f"Supported adducts are: {self.SUPPORTED_ADDUCT_TYPES}"
            )
        
        assert isinstance(hydrogen_rearrangement, HydrogenRearrangement), "hydrogen_rearrangement must be an instance of HydrogenRearrangement"

        self._ion_mode = ion_mode
        self._adduct_types = adduct_types
        self._fragment_tree_builder = fragment_tree_builder
        self._hydrogen_rearrangement = hydrogen_rearrangement

    @property
    def ion_mode(self) -> IonMode:
        return self._ion_mode

    @property
    def adduct_types(self) -> Tuple[Adduct]:
        return tuple(self._adduct_types)

    @property
    def hydrogen_rearrangement(self) -> HydrogenRearrangement:
        return self._hydrogen_rearrangement.copy()

    @property
    def tree_max_depth(self) -> int:
        return self._fragment_tree_builder.max_depth    
    
    @property
    def cleavage_pattern_set(self) -> CleavagePatternSet:
        return self._fragment_tree_builder.cleavage_pattern_set.copy()

    @property
    def cleavage_patterns(self) -> List[CleavagePattern]:
        return list(self._fragment_tree_builder.cleavage_pattern_set.patterns)
    
    @property
    def name(self) -> str:
        return self._fragment_tree_builder.cleavage_pattern_set.name

    def to_dict(self) -> Dict[str, str]:
        """
        Convert this Fragmenter instance into a serializable dictionary.
        """
        return {
            "ion_mode": self._ion_mode.value,
            "fragment_tree_builder": self._fragment_tree_builder.to_dict(),
            "adduct_types": [str(adduct) for adduct in self._adduct_types],
            "hydrogen_rearrangement": self._hydrogen_rearrangement.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Fragmenter":
        """
        Reconstruct a Fragmenter instance from a dictionary.
        """
        ion_mode = parse_ion_mode(data["ion_mode"])
        fragment_tree_builder = FragmentTreeBuilder.from_dict(
            data.get("fragment_tree_builder", {})
        )
        adduct_types = tuple(Adduct.parse(adduct_str) for adduct_str in data.get("adduct_types", []))
        hydrogen_rearrangement = HydrogenRearrangement.from_dict(
            data.get("hydrogen_rearrangement", {})
        )   
        return cls(ion_mode=ion_mode, fragment_tree_builder=fragment_tree_builder, adduct_types=adduct_types, hydrogen_rearrangement=hydrogen_rearrangement)

    def to_yaml(self, path: str):
        """Save the library to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True) 
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )

    @classmethod
    def from_yaml(cls, path: str) -> "Fragmenter":
        """Load a library from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def cleave_all(self, compound: Compound) -> Tuple[FragmentResult]:
        return self._fragment_tree_builder.cleave_all(compound)

    def create_fragment_tree(
            self, 
            compound: Compound, 
            timeout_seconds: float = float('inf'),
            print_info: bool = False,
            ) -> FragmentTree:
        fragment_tree = self._fragment_tree_builder.create_fragment_tree(
            compound,
            timeout_seconds=timeout_seconds,
            print_info=print_info,
        )
        return fragment_tree
    
    def assign_hydrogen_rearrangements_to_fragment_tree(
            self,
            fragment_tree: FragmentTree,
    ) -> 'HydrogenRearrangedFragmentTree':
        return HydrogenRearrangedFragmentTree.from_hydrogen_rearrangement_rules(fragment_tree, self._hydrogen_rearrangement)
    
    def create_hydrogen_rearranged_fragment_tree(
            self, 
            compound: Compound, 
            timeout_seconds: float = float('inf'),
            print_info: bool = False,
            ) -> 'HydrogenRearrangedFragmentTree':
        fragment_tree = self.create_fragment_tree(
            compound,
            timeout_seconds=timeout_seconds,
            print_info=print_info,
        )
        return self.assign_hydrogen_rearrangements_to_fragment_tree(fragment_tree)

    def build_fragment_pathways_by_peak(
            self,
            h_fragment_tree: HydrogenRearrangedFragmentTree,
            precursor_type: Adduct,
            peaks_mz: List[float],
            mass_tolerance: MassTolerance,
    ) -> Tuple[FragmentPathwayGroup, List[FragmentPathwayGroup]]:
        precursor_type_charge = precursor_type.charge

        empty_adduct = Adduct.parse("[M]")
        ion_adduct_strs: Set[str] = set()
        if precursor_type_charge == 1:
            plus_h_adduct = Adduct.parse("[M+H]+")
            normal_adduct: Adduct = Adduct.parse("[M]+")
            minus_h_adduct: Adduct = Adduct.parse("[M-H]+")
            ion_adduct_strs.add(str(plus_h_adduct))

        else:
            raise NotImplementedError("Currently only singly charged precursor adducts are supported.")
        
        adduct_composition, neutral_component_adduct = split_adduct_components(precursor_type, reference_adducts=self.adduct_types)
        assert len(adduct_composition) == 1, "Precursor adduct must contain only one adduct component."
        main_adduct_type = next(iter(adduct_composition))
        assert adduct_composition[main_adduct_type] == 1, "Precursor adduct must contain only one adduct component."
        ion_adduct_strs.add(str(main_adduct_type))

        ion_adducts: Set[Adduct] = {Adduct.parse(adduct_str) for adduct_str in ion_adduct_strs}


        def remove_hs_from_adduct(at: Adduct) -> Adduct:
            at_without_hs = at.copy()
            for f, cnt in at.adduct_formulas.items(): 
                if str(f.plain) == "H":
                    if cnt > 0:
                        r_hs_adduct = Adduct.parse(f"[M-{cnt}H]" if cnt > 1 else "[M-H]")
                    elif cnt < 0:
                        r_hs_adduct = Adduct.parse(f"[M+{abs(cnt)}H]" if abs(cnt) > 1 else "[M+H]")
                    else:
                        continue
                    at_without_hs = at_without_hs.add_prefer_self(r_hs_adduct)
            return at_without_hs
        
        main_adduct_type_without_hs = remove_hs_from_adduct(main_adduct_type)
        
        precursor_compound = Compound.from_smiles(h_fragment_tree.tree.smiles)
        precursor_formula = precursor_type.calc_formula(precursor_compound.formula).normalized
        precursor_type_without_hs = remove_hs_from_adduct(precursor_type)
        precursor_formula_without_hs = precursor_type_without_hs.calc_formula(precursor_compound.formula).normalized
        

        neutral_adduct: Adduct = neutral_component_adduct

        formula_to_node_candidates: Dict[Formula, List[Tuple[int, Compound, Adduct, Formula]]] = defaultdict(list)
        precursor_node_candidates: Set[Tuple[int, str, Adduct]] = set()
        for node_index in range(h_fragment_tree.num_nodes):
            node = h_fragment_tree.tree.get_node(node_index)

            hs_ion_adducts: List[Adduct] = []
            hs_neutral_adducts = h_fragment_tree.neutral_rule_bundle.node_adduct_candidates(node_index)
            if precursor_type_charge == 1:
                ion_hs_candidates = h_fragment_tree.ion_rule_bundle.node_candidates(node_index)
                if 1 in ion_hs_candidates or node_index == 0:
                    hs_ion_adducts.append(plus_h_adduct)
                if 0 in ion_hs_candidates:
                    hs_ion_adducts.append(normal_adduct)
                if -1 in ion_hs_candidates:
                    hs_ion_adducts.append(minus_h_adduct)
                if {1, 0, -1}.issubset(ion_hs_candidates):
                    raise ValueError("Inconsistent hydrogen rearrangement assignments detected.")
            else:
                raise NotImplementedError("Currently only singly charged precursor adducts are supported.")
            
            ion_adduct_pairs: List[Adduct] = list(set(list(ion_adducts)+list(hs_ion_adducts)))
            ion_adduct_with_hs_pairs = list(set([ap.add_prefer_self(na) for ap, na in itertools.product(ion_adduct_pairs, hs_neutral_adducts)]))
            adduct_pairs = list(set([ap.add_prefer_self(na) for ap, na in itertools.product(ion_adduct_with_hs_pairs, [neutral_adduct])]))
            compound = Compound.from_smiles(node.smiles)
            f = compound.formula
            for a in adduct_pairs:
                af = a.calc_formula(f)
                formula_to_node_candidates[af.normalized].append((node_index, compound, a, af))

            for a in ion_adduct_with_hs_pairs:
                af = a.calc_formula(f)
                if af.normalized.value == precursor_formula.normalized.value:
                    precursor_node_candidates.add((node_index, compound.smiles, a))


        precursor_fragment_pathways: List[FragmentPathway] = []
        for node_index, precursor_smiles, at in precursor_node_candidates:
            adduct_type_without_hs = remove_hs_from_adduct(at)
            pathways = FragmentPathway.build_pathways_for_node(h_fragment_tree.tree, node_index, at, precursor_formula, at)
            precursor_fragment_pathways.extend(pathways)
        precursor_fragment_pathway_group = FragmentPathwayGroup.from_list(precursor_fragment_pathways)



        formula_candidates = [f for f in formula_to_node_candidates.keys()]
        assigned_peaks = assign_formulas_to_peaks(
            peaks_mz=peaks_mz,
            formula_candidates=formula_candidates,
            mass_tolerance=mass_tolerance,
        )
        fragment_pathways_by_peak: List[FragmentPathwayGroup] = [FragmentPathwayGroup.from_list([]) for _ in peaks_mz]
        for i, info in enumerate(assigned_peaks):
            fragment_pathways = []
            if info['n_matches'] > 0:
                for formula_str, mass_error in zip(info['matched_formulas'], info['mass_errors']):
                    formula = Formula.parse(formula_str, store_raw=False)
                    pathway_terminal_candidates = formula_to_node_candidates[formula]
                    for node_index, compound, at, adducted_formula in pathway_terminal_candidates:
                        adduct_type_without_hs = remove_hs_from_adduct(at)
                        pathways = FragmentPathway.build_pathways_for_node(h_fragment_tree.tree, node_index, at, precursor_formula_without_hs, adduct_type_without_hs)
                        fragment_pathways.extend(pathways)
            fragment_pathways_by_peak[i] = FragmentPathwayGroup.from_list(fragment_pathways)
        return precursor_fragment_pathway_group, fragment_pathways_by_peak
    
    def parse_fragment_pathway_group(self, fragment_pathway_group_str: str) -> 'FragmentPathwayGroup':
        return FragmentPathwayGroup.parse(fragment_pathway_group_str)

    def copy(self) -> 'Fragmenter':
        """
        Create a copy of the current Fragmenter instance.

        Returns:
            Fragmenter: A new instance with the same adduct_types and max_depth.
        """
        return Fragmenter(
            ion_mode=self._ion_mode,
            fragment_tree_builder=self._fragment_tree_builder.copy(),
            adduct_types=self._adduct_types,
            hydrogen_rearrangement=self._hydrogen_rearrangement.copy(),
        )
    


