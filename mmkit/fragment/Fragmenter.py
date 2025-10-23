from typing import List, Tuple, Dict, Union
from collections import defaultdict
from dataclasses import dataclass
import time
from rdkit import Chem
import json
from pathlib import Path

from .CleavagePattern import CleavagePattern
from .CleavagePatternLibrary import CleavagePatternLibrary
from .FragmentTree import FragmentTree, FragmentNode, FragmentEdge
from ..mass.constants import AdductType, IonMode
from ..chem.Compound import Compound, Formula
from .FragmentResult import FragmentResult
from .FragmentPathway import FragmentStep

class Fragmenter:
    def __init__(
            self,
            max_depth: int,
            cleavage_pattern_lib: CleavagePatternLibrary,
            only_add_min_depth: bool = True
            ):
        self.max_depth = max_depth
        self.cleavage_pattern_lib = cleavage_pattern_lib
        self.only_add_min_depth = only_add_min_depth

    def to_dict(self) -> Dict[str, str]:
        """
        Convert this Fragmenter instance into a serializable dictionary.
        """
        return {
            "max_depth": self.max_depth,
            "cleavage_pattern_lib": self.cleavage_pattern_lib.to_dict(),
            "only_add_min_depth": self.only_add_min_depth
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Fragmenter":
        """
        Reconstruct a Fragmenter instance from a dictionary.
        """
        max_depth = int(data.get("max_depth", 1))
        cleavage_pattern_lib = CleavagePatternLibrary.from_dict(
            data.get("cleavage_pattern_lib", {})
        )
        only_add_min_depth = bool(data.get("only_add_min_depth", False))
        return cls(max_depth=max_depth, cleavage_pattern_lib=cleavage_pattern_lib, only_add_min_depth=only_add_min_depth)

    def save_json(self, path: str):
        """Save the library to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "Fragmenter":
        """Load a library from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def cleave_compound(self, compound: Compound) -> Tuple[FragmentResult]:
        fragment_group = []
        for pattern in self.cleavage_pattern_lib.patterns:
            fragment_result = pattern.fragment(compound)
            if fragment_result is not None:
                fragment_group.append(fragment_result)

        return tuple(fragment_group)

    def create_fragment_tree(
            self, 
            compound: Compound, 
            ion_mode: IonMode,
            timeout_seconds: float = float('inf')
            ) -> FragmentTree:
        
        if ion_mode == IonMode.POSITIVE:
            pass
        else:
            raise NotImplementedError("Only positive ion mode is currently supported.")
                        
        start_time = time.time()

        def check_timeout():
            if (time.time() - start_time) > timeout_seconds:
                raise TimeoutError("Fragmentation process timed out.")

        nodes: Dict[int, FragmentNode] = {}
        edges: Dict[Tuple[int, int], FragmentEdge] = {}
        smi_to_node_id: Dict[str, int] = {}
        processed_node_ids = set()
        node_depths = {}


        def get_set_node_idx(smiles:str, depth:int) -> int:
            if smiles in smi_to_node_id:
                return smi_to_node_id[smiles]
            else:
                node_idx = len(nodes)
                nodes[node_idx] = FragmentNode(node_idx, smiles)
                smi_to_node_id[smiles] = node_idx
                node_depths[node_idx] = depth
            return node_idx
        
        def get_set_edge_idx(
                source_node_idx:int,
                target_node_idx:int,
                fragment_step_str:str,
                depth:int,
                ) -> int:
            edge_key = (source_node_idx, target_node_idx)
            if self.only_add_min_depth and depth > node_depths[target_node_idx]:
                return -1
            
            if edge_key in edges:
                edges[edge_key].try_add_fragment_step(fragment_step_str)
                return edge_key
            else:
                edges[edge_key] = FragmentEdge(
                    source_node_idx,
                    target_node_idx,
                    (fragment_step_str,)
                )
            nodes[source_node_idx].add_child(target_node_idx)
            nodes[target_node_idx].add_parent(source_node_idx)

            return edge_key

        def add_data(
                source_smiles:str,
                target_smiles:str,
                fragment_step_str:str,
                depth:int,
                ) -> Tuple[int, int, int]:
            src_node_idx = get_set_node_idx(source_smiles, depth=depth)
            tgt_node_idx = get_set_node_idx(target_smiles, depth=depth)

            edge_idx = get_set_edge_idx(
                src_node_idx,
                tgt_node_idx,
                fragment_step_str,
                depth=depth,
            )
            return src_node_idx, tgt_node_idx, edge_idx
            
            
        root_compound = compound.copy()
        node_idx = get_set_node_idx(root_compound.smiles, depth=0)

        next_node_ids = [node_idx]
        for depth in range(1, self.max_depth + 1):
            if len(next_node_ids) == 0:
                break
            check_timeout()

            new_node_ids = set()
            for node_idx in next_node_ids:
                source_smiles = nodes[node_idx].smiles
                compound = Compound.from_smiles(source_smiles)
                frag_group = self.cleave_compound(compound)
                for frag_result in frag_group:
                    for frag_product in frag_result.products:
                        target_smiles = frag_product.smiles
                        fragment_step = FragmentStep(
                            cleavage_pattern=frag_result.cleavage,
                            react_indices=frag_product.reactant_indices,
                            prod_indices=frag_product.product_indices,
                        )
                        fragment_step_str = str(fragment_step)
                        # product_mapping = CleavagePattern.parse_product_mapping_str(product_mapping_str)
                        src_node_idx, tgt_node_idx, edge_idx = add_data(
                            source_smiles,
                            target_smiles,
                            fragment_step_str,
                            depth=depth,
                        )
                        new_node_ids.add(tgt_node_idx)
                processed_node_ids.add(node_idx)
            next_node_ids = new_node_ids - processed_node_ids
                
        
        fragment_tree = FragmentTree(compound=root_compound, nodes=nodes, edges=edges)
        return fragment_tree
                
    def copy(self) -> 'Fragmenter':
        """
        Create a copy of the current Fragmenter instance.

        Returns:
            Fragmenter: A new instance with the same adduct_types and max_depth.
        """
        return Fragmenter(
            max_depth=self.max_depth,
            cleavage_pattern_lib=self.cleavage_pattern_lib,
        )

