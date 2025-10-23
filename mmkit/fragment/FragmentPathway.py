from typing import List, Tuple, Union
import json
import re

from .CleavagePattern import CleavagePattern
from .AdductedFragmentTree import AdductedFragmentTree, FragmentTree, FragmentNode, FragmentEdge
from ..chem.Formula import Formula
from ..chem.formula_utils import assign_formulas_to_peaks
from ..mass.Tolerance import MassTolerance
from ..mass.constants import AdductType

class FragmentPathway:
    def __init__(self, path: List[Union[str, 'FragmentPathwayEdge']]):
        for i in range(len(path)):
            if i % 2 == 0:
                assert isinstance(path[i], str), f"Expected str at position {i}, got {type(path[i])}"
            else:
                assert isinstance(path[i], FragmentPathwayEdge), f"Expected FragmentPathwayEdge at position {i}, got {type(path[i])}"
        self.path = path

    def __str__(self):
        pathway_strs = [str(p) for p in self.path]
        pathway_json = json.dumps(pathway_strs, ensure_ascii=False)
        return f'FragmentPathway({pathway_json})'
    
    @staticmethod
    def parse(path_str: str) -> 'FragmentPathway':
        pathway_json = re.match(r'FragmentPathway\((.*)\)', path_str)
        if not pathway_json:
            raise ValueError(f"Invalid FragmentPathway string: {path_str}")
        pathway_content = pathway_json.group(1)
        pathway_strs = json.loads(pathway_content)
        
        pathes: List[Union[str, FragmentPathwayEdge]] = []
        for i, p_str in enumerate(pathway_strs):
            if i % 2 == 0:
                pathes.append(p_str)
            else:
                fragment_pathway_edge = FragmentPathwayEdge.parse(p_str)
                pathes.append(fragment_pathway_edge)
        return FragmentPathway(pathes)

    def __repr__(self):
        return f"FragmentPathway(path={self.path})"

    def get_node(self, index: int) -> str:
        return self.path[index * 2]
    
    def get_edge(self, index: int) -> 'FragmentPathwayEdge':
        return self.path[index * 2 + 1]
    
    @staticmethod
    def build_pathways_by_peak(
        adducted_tree: AdductedFragmentTree, 
        adduct_type: AdductType, 
        peaks_mz:List[float], 
        mass_tolerance:MassTolerance,
        ) -> List[List['FragmentPathway']]:
        all_formulas_with_node_id = adducted_tree.get_all_formulas_with_node_id(adduct_type)
        assigned_peaks = assign_formulas_to_peaks(
            peaks_mz=peaks_mz,
            formula_candidates=[v[0] for v in all_formulas_with_node_id.values()],
            mass_tolerance=mass_tolerance,
        )

        fragment_pathways_by_peak: List[List[FragmentPathway]] = []
        for i, info in enumerate(assigned_peaks):
            fragment_pathways = []
            if info['n_matches'] > 0:
                for formula_str, mass_error in zip(info['matched_formulas'], info['mass_errors']):
                    formula = Formula.parse(formula_str)
                    formula_with_node_id = all_formulas_with_node_id[formula_str]
                    for node_id in formula_with_node_id[1]:
                        pathways = FragmentPathway.build_pathways_for_node(adducted_tree, node_id)
                        fragment_pathways.extend(pathways)
            fragment_pathways_by_peak.append(fragment_pathways)
        return fragment_pathways_by_peak

    @staticmethod
    def build_pathways_for_node(adducted_tree: AdductedFragmentTree, node_id: int) -> List['FragmentPathway']:
        path = FragmentPathway._collect_path_to_root(adducted_tree, node_id)
        fragment_pathways: List[FragmentPathway] = []
        for p in path:
            tmp_path = []
            for i in range(len(p)):
                if i % 2 == 0:
                    tmp_path.append(p[i])
                else:
                    edge_str = p[i]
                    fragment_edge = FragmentEdge.parse(edge_str)
                    cleavage_records = fragment_edge.fragment_step_strs
                    fragment_steps = [FragmentStep.parse(record_str) for record_str in cleavage_records]
                    fragment_pathway_edge = FragmentPathwayEdge(fragment_steps)
                    tmp_path.append(fragment_pathway_edge)
            fragment_pathways.append(FragmentPathway(tmp_path))
        return fragment_pathways

    @staticmethod
    def _collect_path_to_root(tree:Union[AdductedFragmentTree, FragmentTree], node_id: int) -> List[List[str]]:
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
            parent_paths = FragmentPathway._collect_path_to_root(tree, parent_id)

            # Append the current edge and node to each parent path
            for path in parent_paths:
                extended_path = path.copy()
                extended_path.append(str(edge))
                extended_path.append(node.smiles)
                all_paths.append(extended_path)

        return all_paths

class FragmentPathwayEdge:
    def __init__(self, fragment_steps: List['FragmentStep']):
        self.fragment_steps = list(set(fragment_steps))

    def __repr__(self):
        return f"FragmentPathwayEdge(fragment_steps={self.fragment_steps})"
    
    def __str__(self):
        fragment_steps_strs = [str(step) for step in self.fragment_steps]
        fragment_steps_json = json.dumps(fragment_steps_strs, ensure_ascii=False)
        return fragment_steps_json
    
    @staticmethod
    def parse(edge_str: str) -> 'FragmentPathwayEdge':
        try:
            fragment_steps_strs = json.loads(edge_str)
            fragment_steps = [FragmentStep.parse(step_str) for step_str in fragment_steps_strs]
            return FragmentPathwayEdge(fragment_steps)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse FragmentPathwayEdge: {e} → {edge_str}")

class FragmentStep:
    def __init__(self, cleavage_pattern: CleavagePattern, react_indices: Tuple[int,...], prod_indices: Tuple[int,...]):
        self.cleavage_pattern = cleavage_pattern
        self.react_indices = react_indices
        self.prod_indices = prod_indices

    def __str__(self):
        react_json = json.dumps(self.react_indices)
        prod_json = json.dumps(self.prod_indices)
        cleavage_str = str(self.cleavage_pattern)
        cleavage_json = json.dumps(cleavage_str)
        fragment_step_str = json.dumps((react_json, prod_json, cleavage_json))
        return fragment_step_str
    
    @staticmethod
    def parse(step_str: str) -> 'FragmentStep':
        try:
            react_json, prod_json, cleavage_json = json.loads(step_str)
            react_indices = tuple(json.loads(react_json))
            prod_indices = tuple(json.loads(prod_json))
            cleavage_str = json.loads(cleavage_json)
            cleavage_pattern = CleavagePattern.parse(cleavage_str)
            return FragmentStep(cleavage_pattern, react_indices, prod_indices)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse FragmentStep: {e} → {step_str}")
