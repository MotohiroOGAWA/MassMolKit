import os
from typing import Union, List, Dict, Tuple, Literal
from collections import defaultdict
import dill
import copy
from enum import Enum
import networkx as nx

from ..chem.Compound import Compound
from ..mass.Adduct import Adduct
from ..mass.AdductedCompound import AdductedCompound
from ..chem.Formula import Formula
from .BondPosition import BondPosition


class FragmentTree:
    """
    FragmentTree class to represent a tree of fragments.
    """
    class StructureType(Enum):
        """
        Enum to represent the structure type of the fragment tree.
        """
        ORIGINAL = "original"
        TOPOLOGICAL = "topological"

    def __init__(self, compound: Compound, nodes: List['FragmentNode'], edges: List['FragmentEdge']):
        self.compound = compound
        self.nodes: List[FragmentNode] = nodes
        self.edges: List[FragmentEdge] = edges

    def __repr__(self):
        return f"FragmentTree(compound={self.compound.smiles}, nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    @staticmethod
    def empty(compound: Compound) -> 'FragmentTree':
        """
        Create an empty FragmentTree with only the root node.
        """
        return FragmentTree(
            compound=compound,
            nodes=[],
            edges=[]
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

class FragmentNode:
    """
    FragmentNode class to represent a node in the fragment tree.
    """
    def __init__(self, id:int, smiles: str):
        self.id = id
        self.smiles = smiles

    def __repr__(self):
        return f"FragmentNode(id={self.id}, smiles={self.smiles})"

    def __str__(self):
        return f"(id={self.id}, {self.smiles})"

    def copy(self) -> 'FragmentNode':
        """
        Create a copy of the FragmentNode.
        """
        return FragmentNode(
            id=self.id,
            smiles=self.smiles,
        )

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
        
        # Separate "id=1" and the rest
        try:
            id_part, rest = text.split(",", 1)
            node_id = int(id_part.replace("id=", "").strip())
        except Exception as e:
            raise ValueError(f"Failed to parse id: {e} → {text}")

        smiles = rest.strip()

        return FragmentNode(node_id, smiles)

    @staticmethod
    def header() -> str:
        """
        Return the header for the TSV representation of FragmentNode.
        """
        return "ID\tSMILES\n"

    def to_tsv(self):
        """
        Convert the FragmentNode to a TSV string.
        """
        return f"{self.id}\t{self.smiles}\n"

class FragmentEdge:
    def __init__(
        self,
        source_id: int,
        target_id: int,
        cleavage_records: Tuple[str],
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.cleavage_records = tuple(sorted(cleavage_records))

    def __eq__(self, other):
        if not isinstance(other, FragmentEdge):
            raise TypeError(f"Cannot compare FragmentEdge with {type(other).__name__}")
        return (
            self.source_id == other.source_id and
            self.target_id == other.target_id and
            self.cleavage_records == other.cleavage_records
        )

    def __hash__(self):
        return hash((
            self.source_id,
            self.target_id,
            frozenset(self.cleavage_records)
        ))

    def __repr__(self):
        return (
            f"FragmentEdge({self.source_id} -> {self.target_id}, "
            f"cleavage_records={self.cleavage_records})"
        )
    
    def copy(self) -> 'FragmentEdge':
        """
        Create a copy of the FragmentEdge.
        """
        return FragmentEdge(
            source_id=self.source_id,
            target_id=self.target_id,
            cleavage_records=tuple(self.cleavage_records)
        )
    
    def try_add_cleavage_record(self, record: str) -> bool:
        """
        Try to add a cleavage record to the edge.
        Returns True if added, False if already present.
        """
        if record in self.cleavage_records:
            return False
        else:
            self.cleavage_records = tuple(sorted(self.cleavage_records + (record,)))
            return True
    
    @staticmethod
    def header() -> str:
        """
        Return the header for the TSV representation of FragmentEdge.
        """
        return f"Source\tTarget\tCleavages\n"

    def to_tsv(self) -> str:
        """
        Convert the FragmentEdge to a TSV string.
        """
        return f"{self.source_id}\t{self.target_id}\t{str(self.cleavage_records)}\n"

