import os
from typing import Union, List, Dict, Tuple, Literal, Set, Iterable, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import dill
import json
import h5py
import re
from enum import Enum

from ..chem.Compound import Compound
from ..chem.Formula import Formula

PathItem = Union["FragmentNode", "FragmentEdge"]

@dataclass(frozen=True, init=False)
class FragmentTree:
    """
    FragmentTree class to represent a tree of fragments.

    Note:
        frozen=True prevents attribute reassignment, but does NOT prevent
        mutation of mutable objects (e.g., numpy arrays or dict contents).
    """

    _smiles: str
    _node_smiles: np.ndarray            # [N]
    _edge_index: np.ndarray             # [E, 2]
    _edge_step_flat: np.ndarray         # [K]
    _edge_step_indptr: np.ndarray       # [E+1]

    _in_edge_ids: Optional[np.ndarray] = field(default=None, repr=False)
    _in_edge_indptr: Optional[np.ndarray] = field(default=None, repr=False)
    _out_edge_ids: Optional[np.ndarray] = field(default=None, repr=False)
    _out_edge_indptr: Optional[np.ndarray] = field(default=None, repr=False)

    _node_depth: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # Per-instance cache (NEVER use {} as a default)
    _path_cache: Dict[
        Tuple[int, int, Optional[int]],
        List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]
    ] = field(default_factory=dict, repr=False, compare=False)

    def __init__(
        self,
        smiles: str,
        node_smiles: np.ndarray,            # [N]
        edge_index: np.ndarray,             # [E, 2]
        edge_step_flat: np.ndarray,         # [K]
        edge_step_indptr: np.ndarray,       # [E+1]
        in_edge_ids: Optional[np.ndarray] = None,
        in_edge_indptr: Optional[np.ndarray] = None,
        out_edge_ids: Optional[np.ndarray] = None,
        out_edge_indptr: Optional[np.ndarray] = None,
        node_depth: Optional[np.ndarray] = None,
        path_cache: Optional[
            Dict[Tuple[int, Optional[int]], List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]]
        ] = None,
    ):
        # --- Validate inputs ---
        assert node_smiles.ndim == 1, "node_smiles must be a 1D array"
        assert edge_index.ndim == 2 and edge_index.shape[1] == 2, "edge_index must have shape [E, 2]"
        assert edge_step_indptr.ndim == 1 and edge_step_indptr.shape[0] == edge_index.shape[0] + 1, \
            "edge_step_indptr must be length E+1"
        assert edge_step_flat.ndim == 1, "edge_step_flat must be a 1D array"
        assert edge_step_indptr[-1] == len(edge_step_flat), \
            "edge_step_indptr last element must equal len(edge_step_flat)"

        # --- Assign (required for frozen dataclass) ---
        object.__setattr__(self, "_smiles", str(smiles))
        object.__setattr__(self, "_node_smiles", node_smiles)
        object.__setattr__(self, "_edge_index", edge_index)
        object.__setattr__(self, "_edge_step_flat", edge_step_flat)
        object.__setattr__(self, "_edge_step_indptr", edge_step_indptr)

        object.__setattr__(self, "_in_edge_ids", None)
        object.__setattr__(self, "_in_edge_indptr", None)
        object.__setattr__(self, "_out_edge_ids", None)
        object.__setattr__(self, "_out_edge_indptr", None)
        object.__setattr__(self, "_node_depth", node_depth)

        # Optional CSR reuse if valid
        if in_edge_ids is not None and in_edge_indptr is not None:
            if (
                in_edge_indptr.ndim == 1 and
                in_edge_indptr.shape[0] == len(node_smiles) + 1 and
                in_edge_ids.ndim == 1 and
                in_edge_indptr[0] == 0 and
                in_edge_indptr[-1] == len(in_edge_ids)
            ):
                object.__setattr__(self, "_in_edge_ids", in_edge_ids)
                object.__setattr__(self, "_in_edge_indptr", in_edge_indptr)

        if out_edge_ids is not None and out_edge_indptr is not None:
            if (
                out_edge_indptr.ndim == 1 and
                out_edge_indptr.shape[0] == len(node_smiles) + 1 and
                out_edge_ids.ndim == 1 and
                out_edge_indptr[0] == 0 and
                out_edge_indptr[-1] == len(out_edge_ids)
            ):
                object.__setattr__(self, "_out_edge_ids", out_edge_ids)
                object.__setattr__(self, "_out_edge_indptr", out_edge_indptr)

        # Per-instance cache (do not share across instances)
        if path_cache is None:
            object.__setattr__(self, "_path_cache", {})
        else:
            object.__setattr__(self, "_path_cache", path_cache)

    # Optional convenience: safe even under frozen=True (mutates dict contents)
    def clear_path_cache(self) -> None:
        """Clear the internal path cache."""
        self._path_cache.clear()

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
            object.__setattr__(self, "_in_edge_ids", np.asarray([], dtype=np.int32))
            object.__setattr__(self, "_in_edge_indptr", np.zeros(n + 1, dtype=np.int64))
            object.__setattr__(self, "_out_edge_ids", np.asarray([], dtype=np.int32))
            object.__setattr__(self, "_out_edge_indptr", np.zeros(n + 1, dtype=np.int64))
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

        object.__setattr__(self, "_in_edge_ids", in_edge_ids_sorted)
        object.__setattr__(self, "_in_edge_indptr", in_indptr)

        # --- Outgoing edges (group by src) ---
        order_out = np.argsort(src, kind="mergesort")
        src_sorted = src[order_out]
        out_edge_ids_sorted = edge_ids[order_out]

        out_counts = np.bincount(src_sorted, minlength=n).astype(np.int64)
        out_indptr = np.empty(n + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(out_counts, out=out_indptr[1:])

        object.__setattr__(self, "_out_edge_ids", out_edge_ids_sorted)
        object.__setattr__(self, "_out_edge_indptr", out_indptr)

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
    
    def copy(self) -> "FragmentTree":
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
            
            path_cache=None,
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

    def get_root_node_ids(self) -> np.ndarray:
        """Return node ids with no incoming edges."""
        if self._in_edge_ids is None or self._in_edge_indptr is None:
            self._build_edge_adjacency()
        # in-degree = indptr[i+1] - indptr[i]
        indeg = (self._in_edge_indptr[1:] - self._in_edge_indptr[:-1])
        return np.where(indeg == 0)[0].astype(np.int32, copy=False)

    def _build_node_depths(self) -> np.ndarray:
        """
        Compute minimal depth (edge distance) from root(s) to every node.
        Unreachable nodes => -1.
        """
        n = self.num_nodes
        if n == 0:
            depth = np.zeros((0,), dtype=np.int32)
            object.__setattr__(self, "_node_depth", depth)
            return depth

        if self._out_edge_ids is None or self._out_edge_indptr is None or \
        self._in_edge_ids is None or self._in_edge_indptr is None:
            self._build_edge_adjacency()

        roots = self.get_root_node_ids()
        if roots.size == 0:
            roots = np.asarray([0], dtype=np.int32)

        depth = np.full(n, -1, dtype=np.int32)
        q = deque()

        # init
        for r in roots:
            rr = int(r)
            if 0 <= rr < n and depth[rr] == -1:
                depth[rr] = 0
                q.append(rr)

        edge_index = self._edge_index
        out_ids = self._out_edge_ids
        out_ptr = self._out_edge_indptr

        while q:
            u = q.popleft()
            nd = int(depth[u]) + 1

            s = int(out_ptr[u])
            e = int(out_ptr[u + 1])

            # out edge ids are contiguous; iterate without tolist()
            for ei in out_ids[s:e]:
                v = int(edge_index[ei, 1])
                if depth[v] == -1: 
                    depth[v] = nd
                    q.append(v)

        object.__setattr__(self, "_node_depth", depth)

    @property
    def node_depths(self) -> np.ndarray:
        """
        Numpy array [N] of minimal depth from root(s).
        Computed lazily and cached per instance.
        Unreachable nodes => -1.
        """
        if self._node_depth is None:
            self._build_node_depths()
        return self._node_depth

    def get_nodes_by_depth(self) -> Dict[int, np.ndarray]:
        d = self.node_depths
        if d.size == 0:
            return {}
        valid = d >= 0
        if not np.any(valid):
            return {}

        max_d = int(d[valid].max())
        out: Dict[int, np.ndarray] = {}
        for depth in range(max_d + 1):
            ids = np.flatnonzero(d == depth)
            if ids.size:
                out[depth] = ids.astype(np.int32, copy=False)
        return out

    def collect_shortest_paths_to_parents(
        self,
        child_id: int,
        parent_ids: Iterable[int],
        *,
        max_depthes: Optional[Dict[int, Optional[int]]] = None,
        use_cache: bool = True,
    ) -> Dict[int, List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]]:
        """
        Collect ALL shortest path(s) for each parent in parent_ids -> ... -> child_id
        using a single reverse BFS (child -> parents via in_edges).

        Caching policy (IMPORTANT):
        - Cache key is ALWAYS (child_id, parent_id, max_depth) where max_depth is the
            per-parent depth limit (can be None).
        - If a parent is already cached, we do NOT include it in BFS targets.
        - This function merges cached results + newly computed results.

        max_depthes:
            Optional dict {parent_id: max_allowed_depth (edges)}.
            If missing or None: unlimited for that parent.
            Parents not reachable within their limit are returned as empty list (or omitted; see below).
        """

        n = self.num_nodes
        assert 0 <= child_id < n, "Invalid child_id"

        parent_set = {int(pid) for pid in parent_ids}
        if not parent_set:
            return {}

        for pid in parent_set:
            assert 0 <= pid < n, f"Invalid parent_id: {pid}"

        # Normalize per-parent limits (None means unlimited)
        limits: Dict[int, Optional[int]] = {}
        if max_depthes is None:
            for pid in parent_set:
                limits[pid] = None
        else:
            for pid in parent_set:
                limits[pid] = max_depthes.get(pid, None)

        def within_limit(pid: int, d: int) -> bool:
            lim = limits.get(pid, None)
            return True if lim is None else (d <= lim)

        # --- 0) Serve from cache when possible, and build the remaining target set ---
        result: Dict[int, List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]] = {}
        remaining: Set[int] = set()

        if use_cache:
            for pid in parent_set:
                ck = (int(child_id), int(pid), limits[pid])
                if ck in self._path_cache:
                    result[pid] = self._path_cache[ck]
                else:
                    remaining.add(pid)
        else:
            remaining = set(parent_set)

        # If everything was cached, we are done.
        if not remaining:
            return result

        # --- 1) Reverse BFS from child (compute distances upward) ---
        dist = np.full(n, -1, dtype=np.int32)
        dist[child_id] = 0
        q = deque([child_id])

        # nexts[p] contains all (nxt, edge) such that p --edge--> nxt and
        # dist[p] == dist[nxt] + 1 (on some shortest path to child)
        nexts: Dict[int, List[Tuple[int, "FragmentEdge"]]] = defaultdict(list)

        # Track shortest distance for newly requested parents only
        found_dist: Dict[int, int] = {}

        # To stop early safely, we need to know when all remaining parents are either:
        #   - found within limit, OR
        #   - impossible within their limit
        stop_depth: Optional[int] = None

        # Special case: child itself is a requested parent
        if child_id in remaining and within_limit(child_id, 0):
            found_dist[child_id] = 0
            # Note: we still may need BFS for other parents.

        while q:
            x = q.popleft()
            dx = int(dist[x])

            # Safe pruning: once we know stop_depth, no need to expand deeper layers.
            if stop_depth is not None and dx >= stop_depth:
                continue

            # Expand incoming edges
            for e in self.get_in_edges(x):  # e: (p -> x)
                p = int(e.source_id)
                nd = dx + 1

                # Standard BFS discovery
                if dist[p] == -1:
                    dist[p] = nd
                    q.append(p)

                # Record shortest transitions for reconstruction
                if dist[p] == nd:
                    nexts[p].append((x, e))

                # If p is one of the remaining target parents and first time found within limit
                if p in remaining and p not in found_dist and within_limit(p, nd):
                    found_dist[p] = nd

            # Decide if we can stop (only when stop_depth is unknown)
            if stop_depth is None:
                unresolved = []
                for pid in remaining:
                    if pid in found_dist:
                        continue
                    lim = limits.get(pid, None)
                    if lim is None:
                        unresolved.append(pid)  # unlimited and not found yet -> must continue
                    else:
                        # If current frontier depth dx already reached lim, then any new discovery
                        # would have nd=dx+1 > lim -> impossible from now on
                        if dx < lim:
                            unresolved.append(pid)

                if not unresolved:
                    # All remaining parents are either found or impossible within limits.
                    stop_depth = max(found_dist.values()) if found_dist else 0

        # --- 2) Reconstruct ALL shortest paths for each newly found parent ---
        def build_paths_for_parent(pid: int) -> List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]:
            if dist[pid] == -1 or not within_limit(pid, int(dist[pid])):
                return []

            paths: List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]] = []

            def dfs(curr: int, acc: List[Union["FragmentNode", "FragmentEdge"]]) -> None:
                if curr == child_id:
                    paths.append(tuple(acc))
                    return
                for nxt, edge in nexts.get(curr, []):
                    # Stay strictly on shortest layers: dist[curr] == dist[nxt] + 1
                    if dist[curr] == dist[nxt] + 1:
                        dfs(nxt, acc + [edge, self.get_node(nxt)])

            dfs(pid, [self.get_node(pid)])
            return paths

        # Fill results for remaining parents (found or not)
        for pid in remaining:
            if pid == child_id and within_limit(pid, 0):
                paths = [(self.get_node(child_id),)]
            else:
                paths = build_paths_for_parent(pid)

            # Store in output only if non-empty (same behavior as your current code),
            # but we DO cache empty results too, so next call can skip work.
            if paths:
                result[pid] = paths

            if use_cache:
                ck = (int(child_id), int(pid), limits[pid])
                self._path_cache[ck] = paths

        return result

    def collect_shortest_paths_to_root(
        self,
        child_id: int,
        *,
        max_depth: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[Tuple[Union['FragmentNode', 'FragmentEdge'], ...]]:
        """
        Collect all shortest path(s) from root(0) -> ... -> child_id.

        If max_depth is provided, the path length (edges) must be <= max_depth.
        """

        # Use the multi-parent API with a single parent (0).
        max_depthes = None if max_depth is None else {0: int(max_depth)}

        d = self.collect_shortest_paths_to_parents(
            child_id=child_id,
            parent_ids=[0],
            max_depthes=max_depthes,
            use_cache=use_cache,
        )
        return d.get(0, [])
    
    def collect_shortest_paths_via_any_to_root(
        self,
        node_id: int,
        via_node_ids: Iterable[int],
        *,
        max_depth: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]:
        """
        Return ALL globally shortest paths from root(0) to node_id that pass through
        at least one via in via_node_ids (OR condition).

        If max_depth is provided, total edge length must be <= max_depth.

        Returns:
            List of paths (Node, Edge, Node, ...), possibly empty.
        """

        def _edge_len(p: Tuple[Union["FragmentNode", "FragmentEdge"], ...]) -> int:
            return (len(p) - 1) // 2

        def _concat(
            a: Tuple[Union["FragmentNode", "FragmentEdge"], ...],
            b: Tuple[Union["FragmentNode", "FragmentEdge"], ...],
        ) -> Tuple[Union["FragmentNode", "FragmentEdge"], ...]:
            # a: 0 -> ... -> via
            # b: via -> ... -> node
            return a + b[1:]  # drop duplicated via node

        n = self.num_nodes
        assert 0 <= node_id < n, "Invalid node_id"
        assert n > 0, "Empty graph"

        via_set: Set[int] = {int(v) for v in via_node_ids}
        if not via_set:
            return []
        for v in via_set:
            assert 0 <= v < n, f"Invalid via_node_id: {v}"

        depths = self.node_depths  # minimal depth from root(0), unreachable => -1

        # --- Build per-via max depths for (via -> ... -> node) using node_depths ---
        per_via_max_depthes: Optional[Dict[int, Optional[int]]] = None
        valid_vias: List[int] = []

        if max_depth is None:
            for via in via_set:
                if int(depths[via]) >= 0:
                    valid_vias.append(via)
            if not valid_vias:
                return []
        else:
            K = int(max_depth)
            per_via_max_depthes = {}
            for via in via_set:
                dv = int(depths[via])
                if dv < 0:
                    continue
                rem = K - dv
                if rem < 0:
                    continue
                per_via_max_depthes[via] = rem
                valid_vias.append(via)
            if not valid_vias:
                return []

        # --- 1) Collect shortest via->node path sets in ONE BFS ---
        via_to_node: Dict[int, List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]]] = (
            self.collect_shortest_paths_to_parents(
                child_id=node_id,
                parent_ids=valid_vias,
                max_depthes=per_via_max_depthes,
                use_cache=use_cache,
            )
        )
        if not via_to_node:
            return []

        # --- 2) Compute global minimal total length, and gather best vias ---
        min_total: Optional[int] = None
        best_vias: List[int] = []
        via_to_node_len: Dict[int, int] = {}

        for via, paths in via_to_node.items():
            if not paths:
                continue

            dv = int(depths[via])
            if dv < 0:
                continue

            dvn = _edge_len(paths[0])  # all shortest paths share this length
            total = dv + dvn
            if max_depth is not None and total > int(max_depth):
                continue

            via_to_node_len[via] = dvn

            if min_total is None or total < min_total:
                min_total = total
                best_vias = [via]
            elif total == min_total:
                best_vias.append(via)

        if min_total is None:
            return []

        # --- 3) For each best via, get root->via shortest path sets (only those vias) ---
        # Then concatenate ALL combinations to produce ALL globally shortest paths.
        out: List[Tuple[Union["FragmentNode", "FragmentEdge"], ...]] = []

        # Optional: deduplicate if different combos yield identical tuple objects
        seen: Set[str] = set()

        for via in best_vias:
            # root->via shortest paths; cap by depth(via) for efficiency
            root_dict = self.collect_shortest_paths_to_parents(
                child_id=via,
                parent_ids=[0],
                max_depthes={0: int(depths[via])},
                use_cache=use_cache,
            )
            root_paths = root_dict.get(0, [])
            if not root_paths:
                continue

            vn_paths = via_to_node.get(via, [])
            if not vn_paths:
                continue

            # Combine all shortest root->via and all shortest via->node
            for rv in root_paths:
                for vn in vn_paths:
                    full = _concat(rv, vn)

                    # Enforce total length == min_total (and max_depth if given)
                    if _edge_len(full) != int(min_total):
                        continue
                    if max_depth is not None and _edge_len(full) > int(max_depth):
                        continue

                    key = repr(full)
                    if key not in seen:
                        seen.add(key)
                        out.append(full)

        return out

    # def collect_paths_to_root(
    #     self,
    #     node_id: int,
    #     *,
    #     max_depth: Optional[int] = None,
    #     current_depth: int = 0,
    #     use_cache: bool = True,
    # ) -> List[Tuple[Union['FragmentNode', 'FragmentEdge'], ...]]:
    #     """
    #     Collect all possible paths from root to the given node.

    #     Args:
    #         node_id: target node id
    #         max_depth: maximum allowed edge depth
    #         current_depth: internal recursion depth counter
    #         use_cache: whether to use internal memoization cache

    #     Returns:
    #         List of paths ordered from root → node
    #     """

    #     cache_key = (node_id, 0, max_depth)

    #     # Use cache only at top-level call
    #     if use_cache and current_depth == 0 and cache_key in self._path_cache:
    #         return self._path_cache[cache_key]

    #     node = self.get_node(node_id)
    #     in_edges = self.get_in_edges(node_id)

    #     # Root node
    #     if len(in_edges) == 0:
    #         result = [(node,)]
    #         if use_cache and current_depth == 0:
    #             self._path_cache[cache_key] = result
    #         return result

    #     # Depth limit
    #     if max_depth is not None and current_depth > max_depth:
    #         return []

    #     all_paths: List[Tuple[Union[FragmentNode, FragmentEdge], ...]] = []

    #     for in_edge in in_edges:
    #         parent_id = in_edge.source_id

    #         parent_paths = self.collect_paths_to_root(
    #             parent_id,
    #             max_depth=max_depth,
    #             current_depth=current_depth + 1,
    #             use_cache=use_cache,  # propagate flag
    #         )

    #         for path in parent_paths:
    #             extended_path = path + (in_edge, node)
    #             all_paths.append(extended_path)

    #     # Store cache only at top-level
    #     if use_cache and current_depth == 0:
    #         self._path_cache[cache_key] = all_paths

    #     return all_paths

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

