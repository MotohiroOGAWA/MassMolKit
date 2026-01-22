import os
from typing import Union, List, Dict, Tuple, Literal, Set, Optional
from collections import deque, defaultdict
import numpy as np
import dill
import json
import h5py
import re
from enum import Enum

from ..chem.Compound import Compound
from ..chem.Formula import Formula


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

    def __init__(
        self,
        smiles: str,
        node_smiles: np.ndarray,            # [N]
        edge_index: np.ndarray,             # [E, 2]
        edge_step_flat: np.ndarray,         # [K]
        edge_step_indptr: np.ndarray,       # [E+1]
        in_edge_ids: Optional[np.ndarray] = None,      # [Ein]
        in_edge_indptr: Optional[np.ndarray] = None,   # [N+1]
        out_edge_ids: Optional[np.ndarray] = None,     # [Eout]
        out_edge_indptr: Optional[np.ndarray] = None,  # [N+1]
    ):
        assert node_smiles.ndim == 1, "node_smiles must be a 1D array"
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2, "edge_index must be a 2D array with shape [E, 2]"
        assert edge_step_indptr.ndim == 1 and edge_step_indptr.shape[0] == edge_index.shape[0] + 1, "edge_step_indptr must be a 1D array with length E+1"
        assert edge_step_flat.ndim == 1, "edge_step_flat must be a 1D array"

        assert edge_index.shape[0] + 1 == len(edge_step_indptr), "edge_step_indptr length must be E+1"
        assert edge_index.shape[1] == 2, "edge_index must have shape [E, 2]"
        assert edge_step_indptr[-1] == len(edge_step_flat), "edge_step_indptr last element must equal length of edge_step_flat"
        
        self._smiles = smiles
        self._node_smiles = node_smiles
        self._edge_index = edge_index
        self._edge_step_flat = edge_step_flat
        self._edge_step_indptr = edge_step_indptr

        self._in_edge_ids = None
        self._in_edge_indptr = None
        self._out_edge_ids = None
        self._out_edge_indptr = None

        if in_edge_ids is not None and in_edge_indptr is not None:
            if (
                in_edge_indptr.ndim == 1 and
                in_edge_indptr.shape[0] == len(self._node_smiles) + 1 and
                in_edge_ids.ndim == 1 and
                in_edge_indptr[0] == 0 and
                in_edge_indptr[-1] == len(in_edge_ids)
            ):
                self._in_edge_ids = in_edge_ids
                self._in_edge_indptr = in_edge_indptr

        if out_edge_ids is not None and out_edge_indptr is not None:
            if (
                out_edge_indptr.ndim == 1 and
                out_edge_indptr.shape[0] == len(self._node_smiles) + 1 and
                out_edge_ids.ndim == 1 and
                out_edge_indptr[0] == 0 and
                out_edge_indptr[-1] == len(out_edge_ids)
            ):
                self._out_edge_ids = out_edge_ids
                self._out_edge_indptr = out_edge_indptr

    def __repr__(self):
        return f"FragmentTree(compound={self._smiles}, nodes={self.num_nodes}, edges={self.num_edges})"
    
    # ------------------------------------------------------------
    # Edge adjacency builders (single builder)
    # ------------------------------------------------------------
    def _build_edge_adjacency(self) -> None:
        """
        Build both incoming-edge and outgoing-edge adjacency in CSR form.

        Incoming:
          For each node v:
            in_edge_ids[in_edge_indptr[v] : in_edge_indptr[v+1]]
            are edge IDs e such that edge_index[e] = (u -> v).

        Outgoing:
          For each node u:
            out_edge_ids[out_edge_indptr[u] : out_edge_indptr[u+1]]
            are edge IDs e such that edge_index[e] = (u -> v).
        """
        n = self.num_nodes
        e = self.num_edges

        if e == 0:
            self._in_edge_ids = np.asarray([], dtype=np.int32)
            self._in_edge_indptr = np.zeros(n + 1, dtype=np.int64)
            self._out_edge_ids = np.asarray([], dtype=np.int32)
            self._out_edge_indptr = np.zeros(n + 1, dtype=np.int64)
            return

        src = self._edge_index[:, 0].astype(np.int32, copy=False)
        dst = self._edge_index[:, 1].astype(np.int32, copy=False)
        edge_ids = np.arange(e, dtype=np.int32)

        # --- Incoming edges (group by dst) ---
        order_in = np.argsort(dst, kind="mergesort")
        dst_sorted = dst[order_in]
        in_edge_ids_sorted = edge_ids[order_in]

        in_counts = np.bincount(dst_sorted, minlength=n).astype(np.int64)
        in_indptr = np.empty(n + 1, dtype=np.int64)
        in_indptr[0] = 0
        np.cumsum(in_counts, out=in_indptr[1:])

        self._in_edge_ids = in_edge_ids_sorted
        self._in_edge_indptr = in_indptr

        # --- Outgoing edges (group by src) ---
        order_out = np.argsort(src, kind="mergesort")
        src_sorted = src[order_out]
        out_edge_ids_sorted = edge_ids[order_out]

        out_counts = np.bincount(src_sorted, minlength=n).astype(np.int64)
        out_indptr = np.empty(n + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(out_counts, out=out_indptr[1:])

        self._out_edge_ids = out_edge_ids_sorted
        self._out_edge_indptr = out_indptr

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
    def smiles(self) -> str:
        """
        Get the SMILES of the root compound.
        """
        return self._smiles
    
    def get_node(self, node_id: int) -> 'FragmentNode':
        """
        Get the FragmentNode by its ID.
        """
        assert 0 <= node_id < len(self._node_smiles), "Invalid node ID"
        smiles = self._node_smiles[node_id]
        return FragmentNode(id=int(node_id), smiles=smiles)
    
    def get_edges(self, edge_id: int) -> 'FragmentEdge':
        """
        Get the FragmentEdge by its ID.
        """
        assert 0 <= edge_id < self._edge_index.shape[0], "Invalid edge ID"
        source_id = int(self._edge_index[edge_id, 0])
        target_id = int(self._edge_index[edge_id, 1])
        start_idx = self._edge_step_indptr[edge_id]
        end_idx = self._edge_step_indptr[edge_id + 1]
        fragment_step_strs = tuple(self._edge_step_flat[start_idx:end_idx])
        return FragmentEdge(
            source_id=source_id,
            target_id=target_id,
            fragment_step_strs=fragment_step_strs
        )
    
    @property
    def num_nodes(self) -> int:
        """
        Get the number of nodes in the fragment tree.
        """
        return len(self._node_smiles)
    
    @property
    def num_edges(self) -> int:
        """
        Get the number of edges in the fragment tree.
        """
        return self._edge_index.shape[0]
    

    @property
    def in_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._in_edge_ids is None or self._in_edge_indptr is None:
            self._build_edge_adjacency()
        return self._in_edge_ids, self._in_edge_indptr

    @property
    def out_edge_csr(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._out_edge_ids is None or self._out_edge_indptr is None:
            self._build_edge_adjacency()
        return self._out_edge_ids, self._out_edge_indptr

    def get_in_edges(self, node_id: int) -> List['FragmentEdge']:
        """Return incoming edges of node_id."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID"
        if self._in_edge_ids is None or self._in_edge_indptr is None:
            self._build_edge_adjacency()

        s = int(self._in_edge_indptr[node_id])
        e = int(self._in_edge_indptr[node_id + 1])
        in_edge_ids = self._in_edge_ids[s:e]
        return [self.get_edges(int(eid)) for eid in in_edge_ids]
    
    def get_out_edges(self, node_id: int) -> List['FragmentEdge']:
        """Return outgoing edges of node_id."""
        assert 0 <= node_id < self.num_nodes, "Invalid node ID"
        if self._out_edge_ids is None or self._out_edge_indptr is None:
            self._build_edge_adjacency()

        s = int(self._out_edge_indptr[node_id])
        e = int(self._out_edge_indptr[node_id + 1])
        out_edge_ids = self._out_edge_ids[s:e]
        return [self.get_edges(int(eid)) for eid in out_edge_ids]

    def get_parent_nodes(self, child_node_id: int) -> List["FragmentNode"]:
        """Return parent nodes of child_node_id (using incoming edges)."""
        assert 0 <= child_node_id < self.num_nodes, "Invalid node ID"
        if self._in_edge_ids is None or self._in_edge_indptr is None:
            self._build_edge_adjacency()

        s = int(self._in_edge_indptr[child_node_id])
        e = int(self._in_edge_indptr[child_node_id + 1])
        in_edge_ids = self._in_edge_ids[s:e]
        parent_node_ids = self._edge_index[in_edge_ids, 0]
        return [self.get_node(int(pid)) for pid in parent_node_ids]

    def get_child_nodes(self, parent_node_id: int) -> List["FragmentNode"]:
        """Return child nodes of parent_node_id (using outgoing edges)."""
        assert 0 <= parent_node_id < self.num_nodes, "Invalid node ID"
        if self._out_edge_ids is None or self._out_edge_indptr is None:
            self._build_edge_adjacency()

        s = int(self._out_edge_indptr[parent_node_id])
        e = int(self._out_edge_indptr[parent_node_id + 1])
        out_edge_ids = self._out_edge_ids[s:e]
        child_node_ids = self._edge_index[out_edge_ids, 1]
        return [self.get_node(int(cid)) for cid in child_node_ids]
    
    def copy(self) -> 'FragmentTree':
        """
        Create a copy of the FragmentTree.
        """
        return FragmentTree(
            smiles=self._smiles,
            node_smiles=self._node_smiles.copy(),
            edge_index=self._edge_index.copy(),
            edge_step_flat=self._edge_step_flat.copy(),
            edge_step_indptr=self._edge_step_indptr.copy(),
            in_edge_ids=self._in_edge_ids.copy() if self._in_edge_ids is not None else None,
            in_edge_indptr=self._in_edge_indptr.copy() if self._in_edge_indptr is not None else None,
            out_edge_ids=self._out_edge_ids.copy() if self._out_edge_ids is not None else None,
            out_edge_indptr=self._out_edge_indptr.copy() if self._out_edge_indptr is not None else None,
        )
    
    @staticmethod
    def from_nodes_and_edges(smiles:str, nodes: Tuple['FragmentNode'], edges: Tuple['FragmentEdge']) -> 'FragmentTree':
        """
        Create a FragmentTree from lists of FragmentNodes and FragmentEdges.
        """
        node_smiles = np.array([node.smiles for node in nodes], dtype=object)

        edge_index = np.zeros((len(edges), 2), dtype=np.int32)
        edge_step_flat_list = []
        edge_step_indptr = np.zeros(len(edges) + 1, dtype=np.int64)

        for i, edge in enumerate(edges):
            edge_index[i, 0] = edge.source_id
            edge_index[i, 1] = edge.target_id

            start_idx = edge_step_indptr[i]
            step_count = len(edge.fragment_step_strs)
            edge_step_flat_list.extend(edge.fragment_step_strs)
            edge_step_indptr[i + 1] = start_idx + step_count

        edge_step_flat = np.array(edge_step_flat_list, dtype=object)

        return FragmentTree(
            smiles=smiles,
            node_smiles=node_smiles,
            edge_index=edge_index,
            edge_step_flat=edge_step_flat,
            edge_step_indptr=edge_step_indptr
        )

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

        for idx in range(self.num_nodes):
            node = self.get_node(idx)
            compound = Compound.from_smiles(node.smiles)
            formula = compound.formula

            if formula not in self._formula_index_map:
                self._formula_index_map[formula] = []

            self._formula_index_map[formula].append(idx)

    def get_nodes_by_depth(self) -> Dict[int, List[int]]:
        """
        Return a mapping: depth -> list of node IDs.
        Depth starts at 0 for the root node(s).
        If multiple paths reach a node, the minimum depth is used.
        """

        if self.num_nodes == 0:
            return {}

        # --- 1) Find root nodes (nodes without parents) ---
        root_id = 0

        # --- 2) Initialize BFS ---
        queue = deque()
        depth_map = {}  # node_id -> depth

        # Initialize all roots as depth=0
        queue.append((root_id, 0))

        # --- 3) BFS to compute minimal depth of each node ---
        while queue:
            node_id, depth = queue.popleft()

            node = self.get_node(node_id)
            for child_node in self.get_child_nodes(node_id):
                child_id = child_node.id
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

    # ----------------------------
    # HDF5 helpers
    # ----------------------------
    @staticmethod
    def _ensure_utf8_str(x) -> str:
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8")
        return str(x)

    @staticmethod
    def _decode_str_array(a: np.ndarray) -> np.ndarray:
        # robust decode for vlen strings
        out = []
        for x in a:
            out.append(FragmentTree._ensure_utf8_str(x))
        return np.asarray(out, dtype=object)

    @staticmethod
    def _read_tree_id_list(meta_grp) -> list[str]:
        if "tree_ids" not in meta_grp:
            return []
        arr = meta_grp["tree_ids"][:]
        return [FragmentTree._ensure_utf8_str(x) for x in arr]

    @staticmethod
    def _write_tree_id_list(meta_grp, tree_ids: list[str]) -> None:
        dt = h5py.string_dtype(encoding="utf-8")
        # overwrite dataset
        if "tree_ids" in meta_grp:
            del meta_grp["tree_ids"]
        meta_grp.create_dataset("tree_ids", data=np.asarray(tree_ids, dtype=object), dtype=dt)

    @staticmethod
    def _sync_metadata_tree_ids(f: h5py.File) -> None:
        """
        Make metadata/tree_ids consistent with actual groups under /trees.
        """
        meta_grp = f.require_group("metadata")
        trees_grp = f.require_group("trees")

        actual_ids = sorted(list(trees_grp.keys()))
        FragmentTree._write_tree_id_list(meta_grp, actual_ids)

    # ----------------------------
    # Save / Load
    # ----------------------------
    def to_hdf5(
        self,
        path: str,
        tree_id: str,
        mode: Literal["w", "a"] = "w",
    ) -> str:
        """
        Save this FragmentTree into an HDF5 file under /trees/<tree_id>.

        - tree_id is user-specified
        - If /trees/<tree_id> exists, overwrite it.
        - metadata/tree_ids is updated to reflect stored tree_ids.

        Args:
            path: HDF5 file path.
            tree_id: identifier of this tree (group name under /trees).
            mode:
              - 'w': overwrite whole file
              - 'a': append/update within existing file

        Returns:
            tree_id actually saved (same as input)
        """
        assert mode in ("w", "a"), "mode must be 'w' or 'a'"
        tree_id = str(tree_id)

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # normalize dtypes for saving
        node_smiles = np.asarray(self._node_smiles, dtype=object)
        edge_index = np.asarray(self._edge_index, dtype=np.int32)
        edge_step_flat = np.asarray(self._edge_step_flat, dtype=object)
        edge_step_indptr = np.asarray(self._edge_step_indptr, dtype=np.int64)

        dt_str = h5py.string_dtype(encoding="utf-8")

        with h5py.File(path, mode) as f:
            meta_grp = f.require_group("metadata")
            meta_grp.attrs["format"] = "FragmentTree"
            meta_grp.attrs["version"] = "2"  # bump because layout changed (/trees/<tree_id>)

            trees_grp = f.require_group("trees")

            # overwrite if exists
            if tree_id in trees_grp:
                del trees_grp[tree_id]
            grp = trees_grp.create_group(tree_id)

            # store root smiles as attribute
            grp.attrs["smiles"] = str(self._smiles)

            # core datasets
            grp.create_dataset("node_smiles", data=node_smiles, dtype=dt_str)
            grp.create_dataset("edge_index", data=edge_index, compression="gzip")
            grp.create_dataset("edge_step_flat", data=edge_step_flat, dtype=dt_str)
            grp.create_dataset("edge_step_indptr", data=edge_step_indptr, compression="gzip")

            # update metadata list (authoritative from actual groups)
            FragmentTree._sync_metadata_tree_ids(f)

        return tree_id

    @staticmethod
    def from_hdf5(path: str, tree_id: str) -> "FragmentTree":
        """
        Load FragmentTree from /trees/<tree_id>.
        """
        tree_id = str(tree_id)

        with h5py.File(path, "r") as f:
            if "trees" not in f:
                raise ValueError(f"No '/trees' group found in {path}.")
            trees_grp = f["trees"]
            if tree_id not in trees_grp:
                # show available ids (prefer metadata if present)
                available = []
                if "metadata" in f and "tree_ids" in f["metadata"]:
                    available = [FragmentTree._ensure_utf8_str(x) for x in f["metadata"]["tree_ids"][:]]
                else:
                    available = list(trees_grp.keys())
                raise ValueError(f"tree_id '{tree_id}' not found. Available: {sorted(available)}")

            grp = trees_grp[tree_id]

            smiles = FragmentTree._ensure_utf8_str(grp.attrs.get("smiles", ""))

            node_smiles = FragmentTree._decode_str_array(np.asarray(grp["node_smiles"][:], dtype=object))
            edge_index = np.asarray(grp["edge_index"][:], dtype=np.int32)
            edge_step_flat = FragmentTree._decode_str_array(np.asarray(grp["edge_step_flat"][:], dtype=object))
            edge_step_indptr = np.asarray(grp["edge_step_indptr"][:], dtype=np.int64)

            return FragmentTree(
                smiles=smiles,
                node_smiles=node_smiles,
                edge_index=edge_index,
                edge_step_flat=edge_step_flat,
                edge_step_indptr=edge_step_indptr,
            )

    @staticmethod
    def list_tree_ids(path: str) -> Tuple[str, ...]:
        """
        List stored tree_ids. Prefer /metadata/tree_ids if available.
        """
        with h5py.File(path, "r") as f:
            if "metadata" in f and "tree_ids" in f["metadata"]:
                ids = [FragmentTree._ensure_utf8_str(x) for x in f["metadata"]["tree_ids"][:]]
                return tuple(ids)
            if "trees" not in f:
                return tuple()
            return tuple(sorted(list(f["trees"].keys())))

    @staticmethod
    def rebuild_metadata_tree_ids(path: str) -> None:
        """
        Repair metadata/tree_ids to match actual /trees/* groups.
        """
        with h5py.File(path, "a") as f:
            FragmentTree._sync_metadata_tree_ids(f)

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
        return f"(id={self.id};{self.smiles})"

    def copy(self) -> 'FragmentNode':
        """
        Create a copy of the FragmentNode.
        """
        return FragmentNode(
            id=self.id,
            smiles=self.smiles,
        )

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
        fragment_step_strs: Tuple[str],
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.fragment_step_strs = tuple(sorted(set(fragment_step_strs)))

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

