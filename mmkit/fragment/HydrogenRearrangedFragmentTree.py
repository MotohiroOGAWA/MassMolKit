from collections import defaultdict
import numpy as np
from enum import Enum
from typing import Optional
import itertools

from .FragmentTree import FragmentTree
from .HydrogenRearrangement import *
from ..mass.Adduct import Adduct
from ..chem.Formula import Formula

@dataclass(frozen=True)
class IonShiftRuleBundle:
    """
    Bundle of ion-dependent shift rule outputs for a FragmentTree.

    This bundle represents shifts that depend on precursor/adduct type
    (e.g., precursor_type / adduct), so it is typically computed later.

    Fields
    ------
    rule_name:
        Identifier of the ion shift rule (e.g., "ion_shift").
    delta_h_components:
        Per-candidate ΔH components matrix [M, R] or [M, 1].
        - If you have only one ion shift rule, use shape [M, 1].
        - If you may have multiple ion shift rules later, use [M, R].
    node_candidate_indptr:
        CSR-like indptr [N+1] mapping node -> candidate slice in delta_h_components.
    rule_config:
        Serialized configuration of the ion shift rule.
    """
    rule_name: str
    delta_h_components: np.ndarray          # [M], int32
    node_candidate_indptr: np.ndarray       # [N+1], int32
    rule_config: dict

    def __post_init__(self) -> None:
        assert isinstance(self.rule_name, str) and len(self.rule_name) > 0, \
            "rule_name must be a non-empty string"

        assert isinstance(self.delta_h_components, np.ndarray), \
            "delta_h_components must be a numpy array"
        assert self.delta_h_components.ndim == 1, \
            "delta_h_components must be 1D [M]"

        assert isinstance(self.node_candidate_indptr, np.ndarray), \
            "node_candidate_indptr must be a numpy array"
        assert self.node_candidate_indptr.ndim == 1, \
            "node_candidate_indptr must be 1D [N+1]"
        assert self.node_candidate_indptr[0] == 0, \
            "node_candidate_indptr[0] must be 0"
        assert self.node_candidate_indptr[-1] == self.delta_h_components.shape[0], \
            "node_candidate_indptr[-1] must equal M (#rows of delta_h_components)"

        assert isinstance(self.rule_config, dict), "rule_config must be dict"

    @property
    def num_nodes(self) -> int:
        return int(self.node_candidate_indptr.shape[0] - 1)

    @property
    def num_candidates(self) -> int:
        return int(self.delta_h_components.shape[0])

    @property
    def num_rules(self) -> int:
        return int(self.delta_h_components.shape[1])

    def node_candidate_range(self, node_id: int) -> Tuple[int, int]:
        """Return [start, end) candidate row range for the given node."""
        s = int(self.node_candidate_indptr[node_id])
        e = int(self.node_candidate_indptr[node_id + 1])
        return s, e

    def node_candidates(self, node_id: int) -> np.ndarray:
        """Return ΔH component rows [k, R] for the given node."""
        s, e = self.node_candidate_range(node_id)
        return self.delta_h_components[s:e]

    @staticmethod
    def empty(
        rule_name: str,
        num_nodes: int,
        rule_config: Optional[dict] = None,
        num_rules: int = 1,
        dtype=np.int32,
    ) -> "IonShiftRuleBundle":
        """
        Create an empty bundle (no candidates for all nodes).
        Useful when ion-dependent evaluation is postponed.
        """
        if rule_config is None:
            rule_config = {}

        delta_h_components = np.zeros((0, num_rules), dtype=dtype)  # [0, R]
        node_candidate_indptr = np.zeros((num_nodes + 1,), dtype=np.int32)  # all zeros

        return IonShiftRuleBundle(
            rule_name=rule_name,
            delta_h_components=delta_h_components,
            node_candidate_indptr=node_candidate_indptr,
            rule_config=rule_config,
        )

@dataclass(frozen=True)
class HydrogenRearrangementRuleBundle:
    """
    Bundle of hydrogen rearrangement rule outputs for a FragmentTree.

    - rule_names: rule type names in column order of delta_h_components
    - delta_h_components: per-candidate ΔH components matrix [M, R]
    - node_candidate_indptr: CSR-like indptr [N+1] mapping node -> candidate slice
    - rule_configs: serialized configuration for each rule (same order as rule_names)
    """
    rule_names: Tuple[str, ...]             # [R]
    delta_h_components: np.ndarray          # [M, R] int32
    node_candidate_indptr: np.ndarray       # [N+1] int32
    rule_configs: Tuple[dict, ...]          # [R]

    def __post_init__(self) -> None:
        # Basic shape validation
        assert isinstance(self.rule_names, tuple) and len(self.rule_names) > 0, \
            "rule_names must be a non-empty tuple"
        assert isinstance(self.rule_configs, tuple) and len(self.rule_configs) == len(self.rule_names), \
            "rule_configs length must match rule_names length"

        assert isinstance(self.delta_h_components, np.ndarray), "delta_h_components must be a numpy array"
        assert self.delta_h_components.ndim == 2, "delta_h_components must be 2D [M, R]"
        assert self.delta_h_components.shape[1] == len(self.rule_names), \
            "delta_h_components second dimension must match number of rules"

        assert isinstance(self.node_candidate_indptr, np.ndarray), "node_candidate_indptr must be a numpy array"
        assert self.node_candidate_indptr.ndim == 1, "node_candidate_indptr must be 1D [N+1]"
        assert self.node_candidate_indptr[0] == 0, "node_candidate_indptr[0] must be 0"
        assert self.node_candidate_indptr[-1] == self.delta_h_components.shape[0], \
            "node_candidate_indptr[-1] must equal M (#rows of delta_h_components)"

    @property
    def num_rules(self) -> int:
        return len(self.rule_names)

    @property
    def num_candidates(self) -> int:
        return int(self.delta_h_components.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.node_candidate_indptr.shape[0] - 1)

    def node_candidate_range(self, node_id: int) -> Tuple[int, int]:
        """Return [start, end) candidate row range for the given node."""
        s = int(self.node_candidate_indptr[node_id])
        e = int(self.node_candidate_indptr[node_id + 1])
        return s, e

    def node_candidates(self, node_id: int) -> np.ndarray:
        """Return ΔH component rows [k, R] for the given node."""
        s, e = self.node_candidate_range(node_id)
        return self.delta_h_components[s:e]
    
    def node_adduct_candidates(
        self,
        node_id: int,
    ) -> Tuple[Adduct, ...]:
        """Return Adducts corresponding to each candidate for the given node."""
        candidates = self.node_candidates(node_id)  # [k, R]
        adducts: List[Adduct] = []
        for delta_h_components in candidates:
            total_dh = int(np.sum(delta_h_components))
            if total_dh > 0:
                adducts.append(Adduct.parse(f"[M+{total_dh}H]"))  # type: ignore
            elif total_dh < 0:
                adducts.append(Adduct.parse(f"[M{total_dh}H]"))  # type: ignore
            else:
                adducts.append(Adduct.parse(f"[M]"))  # type: ignore
        return tuple(adducts)
    
@dataclass(frozen=True)
class FormulaNodeCandidateIndex:
    """
    CSR-like index for grouping (node_id, candidate_id) pairs by Formula.

    - formulas: sorted by exact mass (ascending)
    - pairs_flat: flattened list of (node_id, candidate_id) for all formulas
    - indptr: CSR pointer of length len(formulas)+1
        pairs for formulas[i] live in pairs_flat[indptr[i]:indptr[i+1]]
    """
    formulas: Tuple[Formula, ...]           # [F]
    pairs_flat: np.ndarray                  # [P, 2] int32 -> (node_id, candidate_id)
    indptr: np.ndarray                      # [F+1] int32

    @staticmethod
    def from_mapping(
        formula_to_node_candidates: Dict[Formula, List[Tuple[int, int]]]
    ) -> "FormulaNodeCandidateIndex":
        """
        Build an index from:
          formula -> list[(node_id, candidate_id)]
        """
        # Sort formulas by exact mass for deterministic ordering
        sorted_formulas = list(formula_to_node_candidates.keys())
        sorted_formulas.sort(key=lambda f: f.exact_mass)

        pairs_flat_list: List[Tuple[int, int]] = []
        formula_pair_indptr: List[int] = [0]

        for formula in sorted_formulas:
            node_candidate_pairs = formula_to_node_candidates[formula]
            node_candidate_pairs.sort()  # sort by (node_id, candidate_id)
            pairs_flat_list.extend(node_candidate_pairs)
            formula_pair_indptr.append(len(pairs_flat_list))

        pairs_flat = np.asarray(pairs_flat_list, dtype=np.int32)  # [P, 2]
        indptr = np.asarray(formula_pair_indptr, dtype=np.int32)  # [F+1]

        return FormulaNodeCandidateIndex(
            formulas=tuple(sorted_formulas),
            pairs_flat=pairs_flat,
            indptr=indptr,
        )

    def get_pairs(self, formula_index: int) -> np.ndarray:
        """Return (node_id, candidate_id) pairs for formulas[formula_index]."""
        s = int(self.indptr[formula_index])
        e = int(self.indptr[formula_index + 1])
        return self.pairs_flat[s:e]
    
class HydrogenRearrangedFragmentTree:
    """
    FragmentTree + hydrogen rearrangement candidates per node.
    Does NOT change topology; only annotates nodes with possible ΔH shifts.
    """
    def __init__(
            self, 
            base_tree: FragmentTree, 
            neutral_rule_bundle: HydrogenRearrangementRuleBundle,
            ion_rule_bundle: IonShiftRuleBundle,
            formula_index: FormulaNodeCandidateIndex,   # grouping by rearranged formula
            ):
        # Validate node count consistency
        assert neutral_rule_bundle.num_nodes == base_tree.num_nodes, \
            ".num_nodes must match base_tree.num_nodes"
        assert ion_rule_bundle.num_nodes == base_tree.num_nodes, \
            ".num_nodes must match base_tree.num_nodes"

        self._base_tree = base_tree
        self._neutral_rule_bundle = neutral_rule_bundle
        self._ion_rule_bundle = ion_rule_bundle
        self._formula_index = formula_index

    @property
    def tree(self) -> FragmentTree:
        return self._base_tree
    
    @property
    def neutral_rule_bundle(self) -> HydrogenRearrangementRuleBundle:
        return self._neutral_rule_bundle

    @property
    def ion_rule_bundle(self) -> IonShiftRuleBundle:
        return self._ion_rule_bundle

    @property
    def smiles(self) -> str:
        return self._base_tree.smiles
    
    @property
    def formula_index(self) -> FormulaNodeCandidateIndex:
        """Return formula -> (node, candidate) grouping index."""
        return self._formula_index
    
    @property
    def num_nodes(self) -> int:
        return self._base_tree.num_nodes
    
    @property
    def num_edges(self) -> int:
        return self._base_tree.num_edges

    @staticmethod
    def from_hydrogen_rearrangement_rules(
        tree: FragmentTree,
        hydrogen_rearrangement: HydrogenRearrangement,
    ) -> "HydrogenRearrangedFragmentTree":
        """
        Construct a HydrogenRearrangedFragmentTree by evaluating hydrogen
        rearrangement rules for each fragment node.

        Parameters
        ----------
        tree : FragmentTree
            Base fragment tree.
        hydrogen_rearrangement : HydrogenRearrangement
            Container of hydrogen rearrangement rules.

        Returns
        -------
        HydrogenRearrangedFragmentTree
        """

        # List of active rearrangement rules
        rearrangement_rules = [
            hydrogen_rearrangement.bond_unsaturation_rule,
            hydrogen_rearrangement.radical_rule,
        ]

        # Corresponding ΔH candidate lists for each rule
        delta_h_candidates_per_rule = [
            rule.delta_h_candidates for rule in rearrangement_rules
        ]

        #
        rearrangement_rule_names = [
            rule.RULE_TYPE for rule in rearrangement_rules
        ]

        # 
        rearrangement_rule_configs = [
            rule.to_dict() for rule in rearrangement_rules
        ]

        ion_rule_delta_h_candidates = hydrogen_rearrangement.ion_shift_rule.delta_h_candidates

        neutral_node_shift_indptr = [0]
        neutral_flat_delta_h: list[tuple[int, ...]] = []
        total_dh_to_adducts: Dict[int, Adduct] = {}
        formula_to_node_candidates: Dict[Formula, List[Tuple[int, ...]]] = defaultdict(list)

        ion_node_shift_indptr = [0]
        ion_flat_delta_h: list[int] = []

        # Iterate over fragment nodes
        for node_index, node_smiles in enumerate(tree._node_smiles):
            compound = Compound.from_smiles(node_smiles)
            formula = compound.formula

            ion_rule_delta_h_mask = hydrogen_rearrangement.ion_shift_rule.evaluate(compound)
            ion_h_shifts = [
                ion_rule_delta_h_candidates[i]
                for i, is_enabled in enumerate(ion_rule_delta_h_mask)
                if is_enabled
            ]
            ion_flat_delta_h.extend(ion_h_shifts)
            ion_node_shift_indptr.append(len(ion_flat_delta_h))


            per_rule_enabled_shifts = []

            # Evaluate each rearrangement rule
            for rule, delta_h_candidates in zip(
                rearrangement_rules, delta_h_candidates_per_rule
            ):
                # rule_mask: boolean mask indicating which candidates are valid
                rule_mask = rule.evaluate(compound)

                # Extract enabled ΔH candidates
                enabled_shifts = [
                    delta_h_candidates[i]
                    for i, is_enabled in enumerate(rule_mask)
                    if is_enabled
                ]

                per_rule_enabled_shifts.append(enabled_shifts)

            # Combine ΔH candidates across rules (cartesian product)
            combined_shifts = list(itertools.product(*per_rule_enabled_shifts))
            for i, s in enumerate(combined_shifts):
                dh = sum(s)
                if dh not in total_dh_to_adducts:
                    if dh > 0:
                        total_dh_to_adducts[dh] = Adduct.parse(f"[M+{dh}H]")  # type: ignore
                    elif dh < 0:
                        total_dh_to_adducts[dh] = Adduct.parse(f"[M{dh}H]")  # type: ignore
                    else:
                        total_dh_to_adducts[dh] = Adduct.parse(f"[M]")  # type: ignore
                
                rearranged_formula = total_dh_to_adducts[dh].calc_formula(formula)
                formula_to_node_candidates[rearranged_formula.normalized].append((node_index, len(neutral_flat_delta_h) + i))


            neutral_flat_delta_h.extend(combined_shifts)
            neutral_node_shift_indptr.append(len(neutral_flat_delta_h))

        delta_h_matrix = np.array(neutral_flat_delta_h, dtype=int)
        delta_h_shift_indptr = np.array(neutral_node_shift_indptr, dtype=int)

        neutral_rule_bundle = HydrogenRearrangementRuleBundle(
            rule_names=tuple(rearrangement_rule_names),
            delta_h_components=delta_h_matrix,
            node_candidate_indptr=delta_h_shift_indptr,
            rule_configs=tuple(rearrangement_rule_configs),
        )

        ion_delta_h_array = np.array(ion_flat_delta_h, dtype=int)
        ion_shift_indptr = np.array(ion_node_shift_indptr, dtype=int)
        ion_rule_bundle = IonShiftRuleBundle(
            rule_name=hydrogen_rearrangement.ion_shift_rule.RULE_TYPE,
            delta_h_components=ion_delta_h_array,  # [M]
            node_candidate_indptr=ion_shift_indptr,
            rule_config=hydrogen_rearrangement.ion_shift_rule.to_dict(),
        )
        
        formula_index = FormulaNodeCandidateIndex.from_mapping(formula_to_node_candidates)

        return HydrogenRearrangedFragmentTree(
            base_tree=tree,
            neutral_rule_bundle=neutral_rule_bundle,
            ion_rule_bundle=ion_rule_bundle,
            formula_index=formula_index,
        )
    
