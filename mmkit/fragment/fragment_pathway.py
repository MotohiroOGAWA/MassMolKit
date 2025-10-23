from typing import Dict, List, Optional, Tuple

from .AdductedFragmentTree import AdductedFragmentTree
from ..mass.constants import AdductType, IonMode
from ..mass.Tolerance import MassTolerance, PpmTolerance, DaTolerance
from ..chem.formula_utils import assign_formulas_to_peaks
from ..chem.Formula import Formula

def assign_fragment_pathway():
    pass

def func1(peaks_mz:List[float]):
    pass

def func3(
        adducted_tree: AdductedFragmentTree, 
        adduct_type:AdductType, 
        peaks_mz:List[float],
        mass_tolerance:MassTolerance,
        ) -> List[List[str]]:
    all_formulas_with_node_id = adducted_tree.get_all_formulas_with_node_id(adduct_type)
    assigned_peaks = assign_formulas_to_peaks(
        peaks_mz=peaks_mz,
        formula_candidates=[v[0] for v in all_formulas_with_node_id.values()],
        mass_tolerance=mass_tolerance,
    )

    for i, info in enumerate(assigned_peaks):
        if info['n_matches'] > 0:
            for formula_str, mass_error in zip(info['matched_formulas'], info['mass_errors']):
                formula = Formula.parse(formula_str)
                formula_with_node_id = all_formulas_with_node_id[formula_str]
                for node_id in formula_with_node_id[1]:
                    paths = _collect_path_to_root(adducted_tree, node_id)
                    return paths

def func2(adducted_tree: AdductedFragmentTree, node_id: int) -> List[str]:
    path = _collect_path_to_root(adducted_tree, node_id)
    return path

def _collect_path_to_root(tree, node_id: int) -> List[List[str]]:
    """
    Recursively collect all possible paths (as lists of str(node) and str(edge))
    from the given node up to the root.
    Returns:
        A list of paths, each path being a list of strings ordered from root → current node.
    """
    node = tree.nodes[node_id]

    # Base case: node has no parents (root)
    if not node.parent_ids:
        return [[node.smiles]]

    all_paths: List[List[str]] = []

    # Explore all parent nodes
    for parent_id in node.parent_ids:
        edge = tree.edges[(parent_id, node_id)]

        # Recursively collect paths from this parent
        parent_paths = _collect_path_to_root(tree, parent_id)

        # Append the current edge and node to each parent path
        for path in parent_paths:
            extended_path = path.copy()
            extended_path.append(str(edge))
            extended_path.append(node.smiles)
            all_paths.append(extended_path)

    return all_paths