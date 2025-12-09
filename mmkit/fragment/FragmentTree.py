import os
from typing import Union, List, Dict, Tuple, Literal, Set
import dill
import json
import copy
import re
from enum import Enum

from ..chem.Compound import Compound
from ..chem.Formula import Formula
from ..mass.Adduct import Adduct, split_adduct_components
from ..mass.constants import IonMode


class FragmentTree:
    """
    FragmentTree class to represent a tree of fragments.
    """
    ADDUCT_EMPTY_POS1 = Adduct.parse('[M]+')
    ADDUCT_M_PLUS_H_POS1 = Adduct.parse('[M+H]+')
    ADDUCT_EMPTY_NEG1 = Adduct.parse('[M]-')
    ADDUCT_M_MINUS_H_NEG1 = Adduct.parse('[M-H]-')

    class StructureType(Enum):
        """
        Enum to represent the structure type of the fragment tree.
        """
        ORIGINAL = "original"
        TOPOLOGICAL = "topological"

    def __init__(self, compound: Compound, nodes: Dict[int, 'FragmentNode'], edges: Dict[Tuple[int, int], 'FragmentEdge']):
        self.compound = compound
        self.nodes: Dict[int, FragmentNode] = nodes
        self.edges: Dict[Tuple[int, int], FragmentEdge] = edges

    def __repr__(self):
        return f"FragmentTree(compound={self.compound.smiles}, nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    @staticmethod
    def empty(compound: Compound) -> 'FragmentTree':
        """
        Create an empty FragmentTree with only the root node.
        """
        return FragmentTree(
            compound=compound,
            nodes={},
            edges={}
        )
    
    @property
    def num_nodes(self) -> int:
        """
        Get the number of nodes in the fragment tree.
        """
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        """
        Get the number of edges in the fragment tree.
        """
        return len(self.edges)
    
    def save(self, file_path: str):
        """
        Save this FragmentTree to a .dill file.
        Args:
            file_path (str): Output path to save the file.
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(self, f, protocol=dill.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: str) -> 'FragmentTree':
        """
        Load a FragmentTree object from a .dill file.
        Args:
            file_path (str): Path to the saved file.
        Returns:
            FragmentTree: Loaded instance.
        """
        with open(file_path, "rb") as f:
            tree = dill.load(f)

        return tree
    
    @property
    def formula_index_map(self) -> Dict[Formula, List[int]]:
        """
        Get the lookup table mapping (Formula) to the list of fragment node indices
        having that formula.
        """
        if not hasattr(self, '_formula_index_map') or self._formula_index_map is None:
            self._build_formula_index_map()
        return self._formula_index_map.copy()
    
    def _build_formula_index_map(self) -> None:
        """
        Build the lookup table mapping (Formula) to the list of fragment node indices
        having that formula.
        """
        self._formula_index_map = {}

        for idx, node in self.nodes.items():
            compound = Compound.from_smiles(node.smiles)
            formula = compound.formula

            if formula not in self._formula_index_map:
                self._formula_index_map[formula] = []

            self._formula_index_map[formula].append(idx)


    def get_all_formulas_with_node_id(self, precursor_type: Adduct, supported_adduct_types: Tuple[Adduct]) -> Dict[str, Tuple[Formula, Adduct, List[int]]]:
        adduct_composition, neutral_component_adduct = split_adduct_components(precursor_type, reference_adducts=supported_adduct_types)
        assert len(adduct_composition) == 1, f"Only one neutral component is supported: {precursor_type}, {adduct_composition}"
        adduct_type = next(iter(adduct_composition))
        assert adduct_composition[adduct_type] == 1, f"Only single charge adducts are supported: {precursor_type}, {adduct_composition}"

        if adduct_type.charge > 0:
            ion_mode = IonMode.POSITIVE
            charge_mode = 1
            adducts = [Adduct.parse(at) for at in set([str(adduct_type), "[M+H]+"])]
            empty_adduct = FragmentTree.ADDUCT_EMPTY_POS1
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
            for formula, node_indices in self.formula_index_map.items():
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

    def get_nodes_by_depth(self) -> Dict[int, List[int]]:
        """
        Return a mapping: depth -> list of node IDs.
        Depth starts at 0 for the root node(s).
        If multiple paths reach a node, the minimum depth is used.
        """

        if not self.nodes:
            return {}

        # --- 1) Find root nodes (nodes without parents) ---
        root_id = 0

        # --- 2) Initialize BFS ---
        from collections import deque, defaultdict

        queue = deque()
        depth_map = {}  # node_id -> depth

        # Initialize all roots as depth=0
        queue.append((root_id, 0))

        # --- 3) BFS to compute minimal depth of each node ---
        while queue:
            node_id, depth = queue.popleft()

            node = self.nodes[node_id]
            for child_id in node.child_ids:

                # If unassigned, assign depth+1
                if child_id not in depth_map:
                    depth_map[child_id] = depth + 1
                    queue.append((child_id, depth + 1))

                # If already assigned, keep the minimum depth
                elif depth + 1 < depth_map[child_id]:
                    depth_map[child_id] = depth + 1
                    queue.append((child_id, depth + 1))

        # --- 4) Convert depth_map → depth -> list[node_id] ---
        depth_to_nodes = defaultdict(list)
        for node_id, depth in depth_map.items():
            depth_to_nodes[depth].append(node_id)

        # Sort each node list for consistency
        return {d: sorted(ids) for d, ids in depth_to_nodes.items()}

class FragmentNode:
    """
    FragmentNode class to represent a node in the fragment tree.
    """
    def __init__(self, id:int, smiles: str, parent_ids: Tuple[int]=(), child_ids: Tuple[int]=()):
        self.id = id
        self.smiles = smiles
        self.parent_ids = tuple(set(parent_ids))
        self.child_ids = tuple(set(child_ids))

    def __repr__(self):
        return f"FragmentNode(id={self.id}, smiles={self.smiles})"

    def __str__(self):
        return f"(id={self.id};{self.smiles};parents={list(self.parent_ids)};children={list(self.child_ids)})"

    def copy(self) -> 'FragmentNode':
        """
        Create a copy of the FragmentNode.
        """
        return FragmentNode(
            id=self.id,
            smiles=self.smiles,
            parent_ids=tuple(self.parent_ids),
            child_ids=tuple(self.child_ids),
        )
    
    def add_parent(self, parent_id: int):
        """
        Add a parent ID to the FragmentNode.
        """
        if parent_id not in self.parent_ids:
            self.parent_ids = tuple(set(self.parent_ids + (parent_id,)))

    def add_child(self, child_id: int):
        """
        Add a child ID to the FragmentNode.
        """
        if child_id not in self.child_ids:
            self.child_ids = tuple(set(self.child_ids + (child_id,)))

    @staticmethod
    def parse(text: str) -> "FragmentNode":
        """
        Parse a string created by __str__() back into a FragmentNode.
        Expected format:
            (id=1, CC(=O)O)
        """
        # --- Cleanup ---
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]  # remove parentheses

        # --- Split off id= ---
        if not text.startswith("id="):
            raise ValueError(f"Invalid FragmentNode string: missing 'id=' prefix → {text}")
        
        parts = [p.strip() for p in text.split(";")]

        # id
        if not parts[0].startswith("id="):
            raise ValueError(f"Invalid format: {text}")
        node_id = int(parts[0].replace("id=", "").strip())

        # smiles
        smiles = parts[1].strip()

        # parents / children
        parent_ids, child_ids = [], []
        for part in parts[2:]:
            if part.startswith("parents="):
                parent_ids = [int(x) for x in part.replace("parents=", "").strip(" []").split(",") if x]
            elif part.startswith("children="):
                child_ids = [int(x) for x in part.replace("children=", "").strip(" []").split(",") if x]

        node = FragmentNode(node_id, smiles, parent_ids, child_ids)
        return node

    @staticmethod
    def header() -> str:
        """
        Return the header for the TSV representation of FragmentNode.
        """
        return "ID\tSMILES\tParents\tChildren\n"

    def to_tsv(self):
        """
        Convert the FragmentNode to a TSV string.
        """
        return f"{self.id}\t{self.smiles}\t{self.parent_ids}\t{self.child_ids}\n"
    


class FragmentEdge:
    def __init__(
        self,
        source_id: int,
        target_id: int,
        fragment_step_strs: Tuple[str],
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.fragment_step_strs = tuple(sorted(fragment_step_strs))

    def __eq__(self, other):
        if not isinstance(other, FragmentEdge):
            raise TypeError(f"Cannot compare FragmentEdge with {type(other).__name__}")
        return (
            self.source_id == other.source_id and
            self.target_id == other.target_id and
            self.fragment_step_strs == other.fragment_step_strs
        )

    def __hash__(self):
        return hash((
            self.source_id,
            self.target_id,
            frozenset(self.fragment_step_strs)
        ))

    def __repr__(self):
        return (
            f"FragmentEdge({self.source_id} -> {self.target_id}, "
            f"fragment_steps={self.fragment_step_strs})"
        )
    
    def __str__(self):
        fragment_steps_json = json.dumps(self.fragment_step_strs, ensure_ascii=False)
        return f"(src={self.source_id}; tgt={self.target_id}; fragment_steps={fragment_steps_json})"

    @staticmethod
    def parse(text: str) -> "FragmentEdge":
        """
        Parse a string created by __str__() back into a FragmentEdge.
        Expected format:
            (src=0; tgt=1; fragment_steps=["C1-C2","C3-O4"])
        """
        text = text.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]

        # --- source id ---
        m_src = re.search(r"src=(\d+)", text)
        m_tgt = re.search(r"tgt=(\d+)", text)
        m_clv = re.search(r"fragment_steps=(\[.*\])", text)

        if not (m_src and m_tgt and m_clv):
            raise ValueError(f"Invalid FragmentEdge string: {text}")

        source_id = int(m_src.group(1))
        target_id = int(m_tgt.group(1))
        fragment_steps_json = m_clv.group(1)

        try:
            fragment_step_strs = tuple(json.loads(fragment_steps_json))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse fragment_steps: {e} → {fragment_steps_json}")

        return FragmentEdge(source_id, target_id, fragment_step_strs)

    def copy(self) -> 'FragmentEdge':
        """
        Create a copy of the FragmentEdge.
        """
        return FragmentEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            fragment_step_strs=tuple(self.fragment_step_strs)
        )
    
    def try_add_fragment_step(self, fragment_step_str: str) -> bool:
        if fragment_step_str in self.fragment_step_strs:
            return False
        else:
            self.fragment_step_strs = tuple(sorted(self.fragment_step_strs + (fragment_step_str,)))
            return True
    
    @staticmethod
    def header() -> str:
        """
        Return the header for the TSV representation of FragmentEdge.
        """
        return f"Source\tTarget\tFragmentSteps\n"

    def to_tsv(self) -> str:
        """
        Convert the FragmentEdge to a TSV string.
        """
        return f"{self.source_id}\t{self.target_id}\t{str(self.fragment_step_strs)}\n"

