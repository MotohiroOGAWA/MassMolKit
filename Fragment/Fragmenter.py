from typing import List, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass
import time
from rdkit import Chem

from .CleavagePattern import CleavagePattern
from .cleavage_patterns import patterns as default_cleavage_patterns
from .FragmentTree import FragmentTree, FragmentNode, FragmentEdge
from ..MS.constants import AdductType
from ..Mol.Compound import Compound
from ..MS.Adduct import Adduct
from ..MS.AdductIon import AdductIon
from ..Fragment.Frag import Frag
from ..Fragment.BondPosition import BondPosition

class Fragmenter:
    _SUPPORTED_ADDUCT_TYPES = [AdductType.M_PLUS_H_POS]
    def __init__(
            self,
            adduct_type: Tuple[AdductType],
            max_depth: int,
            cleavage_patterns: Tuple[CleavagePattern] = None
            ):
        assert all(isinstance(at, AdductType) for at in adduct_type), "adduct_type must be a tuple of AdductType"
        assert all((at in self._SUPPORTED_ADDUCT_TYPES) for at in adduct_type), "Currently only AdductType.M_PLUS_H_POS is supported"

        self.adduct_type = adduct_type
        self.max_depth = max_depth
        self.patterns = cleavage_patterns if cleavage_patterns is not None else default_cleavage_patterns

    def cleave_compound(self, compound: Compound, adduct_type: AdductType) -> Dict[str, Tuple['FragmentInfo']]:
        """
        Cleave the compound using the defined cleavage patterns.

        Parameters:
            compound (Compound): The input compound to be cleaved.

        Returns:
            List[Compound]: A list of resulting fragments as Compound objects.
        """
        fragments = defaultdict(list)
        for pattern in self.patterns:
            if not pattern.exists(compound._mol):
                continue

            frag_groups = pattern.fragment(compound._mol)
            for frag_group in frag_groups:
                for frag_idx, mol in enumerate(frag_group):
                    frag_c = Compound(mol)
                    frag = Frag(frag_c, adduct_type=adduct_type)

                    # Identify bond positions (dummy atoms) in the fragment
                    bond_poses = []
                    for atom in mol.GetAtoms():
                        if atom.GetAtomicNum() == 0:
                            neighbors = [nei.GetAtomMapNum() for nei in atom.GetNeighbors()]
                            if len(neighbors) != 1:
                                raise ValueError("Unexpected number of neighbors for dummy atom.")
                            neighbor_map_num = neighbors[0]
                            bond_poses.append(neighbor_map_num)
                    bond_poses = tuple(sorted(bond_poses))

                    for recon_c, adducts in frag._candidates.items():
                        for adduct in adducts:
                            fragment_info = FragmentInfo(
                                smiles=recon_c._smiles,
                                adduct=adduct,
                                parent_smiles=compound._smiles,
                                bond_positions=bond_poses,
                                cleavage_pattern=(pattern.smirks, frag_idx)
                            )
                            fragments[recon_c.smiles].append(fragment_info)

        # Remove duplicate fragments
        unique_fragments = {}
        for smi, frag_list in fragments.items():
            unique_fragments[smi] = tuple(set(frag_list))
            
        return unique_fragments

    def create_fragment_tree(self, compound: Compound, timeout_seconds: float = float('inf')) -> FragmentTree:
        """`
        Create a fragment tree from the compound.

        Parameters:
            compound (Compound): The input compound.
            timeout_seconds (float): The maximum time allowed for processing. If None, no timeout is applied.

        Raises:
            TimeoutError: Raised if the processing time exceeds the specified timeout.

        Returns:
            FragmentTree: The resulting fragment tree.
        """
        start_time = time.time()

        def check_timeout():
            if (time.time() - start_time) > timeout_seconds:
                raise TimeoutError("Fragmentation process timed out.")

        nodes: List[FragmentNode] = []
        edges: List[FragmentEdge] = []
        smi_to_node_id: Dict[str, int] = {}
        node_id_pair_to_edge_id: Dict[Tuple[int, int], int] = {}

        def add_node(compound: Compound, adducts: Tuple[Adduct]) -> int:
            smi = compound.smiles
            adduct_strs = tuple(str(adduct) for adduct in adducts)
            if smi in smi_to_node_id:
                node_id = smi_to_node_id[smi]
                adduct_strs = tuple(set(nodes[node_id].adducts).union(set(adduct_strs)))
                nodes[node_id].adducts = adduct_strs
            else:
                node_id = len(nodes)
                nodes.append(FragmentNode(smi, adduct_strs))
                smi_to_node_id[smi] = node_id
            return node_id
        
        def add_edges(edge: FragmentEdge):
            key = (edge.source_id, edge.target_id)
            if key in node_id_pair_to_edge_id:
                old_edge_id = node_id_pair_to_edge_id[key]
                old_edge = edges[old_edge_id]
                old_edge.cleavage_records = tuple(set(old_edge.cleavage_records).union(set(edge.cleavage_records)))
                old_edge.attribute.update(edge.attribute)
                edges[old_edge_id] = old_edge
                edge = old_edge
                edge_id = old_edge_id
            else:
                edge_id = len(edges)
                edges.append(edge)
                node_id_pair_to_edge_id[key] = edge_id
            
            return edge_id

        for adduct_type in self.adduct_type:
            if adduct_type == AdductType.M_PLUS_H_POS:
                precursor = Frag(compound, adduct_type=adduct_type)
            else:
                raise NotImplementedError(f"Adduct type {adduct_type} not implemented.")
            
            next_node_ids = []
            processed_node_ids = set()

            for c, adducts in precursor._candidates.items():
                next_node_id = add_node(c, adducts)
                next_node_ids.append(next_node_id)


            for depth in range(1, self.max_depth + 1):
                if len(next_node_ids) == 0:
                    break

                check_timeout()
                new_node_ids = []
                new_infoes: Dict[str, Tuple[FragmentInfo]] = defaultdict(list)
                for node_id in next_node_ids:
                    fragment_node: FragmentNode = nodes[node_id]
                    frag_groups = self.cleave_compound(Compound(Chem.MolFromSmiles(fragment_node.smiles)), adduct_type=adduct_type)
                    for smi, frag_infoes in frag_groups.items():
                        new_infoes[smi].extend(frag_infoes)
                    processed_node_ids.add(node_id)

                for smi, frag_infoes in new_infoes.items():
                    check_timeout()
                    c = Compound.from_smiles(smi)
                    adducts = tuple(frag_info.adduct for frag_info in frag_infoes)
                    new_node_id = add_node(c, adducts)
                    if new_node_id not in processed_node_ids:
                        new_node_ids.append(new_node_id)

                    for frag_info in frag_infoes:
                        source_c = Compound.from_smiles(frag_info.parent_smiles)
                        _atom_map_to_idx = source_c.atom_map_to_idx
                        bond_pos_idx = tuple(sorted(_atom_map_to_idx[bond_pos] for bond_pos in frag_info.bond_positions))
                        source_id = smi_to_node_id[source_c.smiles]

                        cleavage_record = (frag_info.cleavage_pattern[0], frag_info.cleavage_pattern[1], bond_pos_idx)
                        edge = FragmentEdge(
                            source_id=source_id,
                            target_id=new_node_id,
                            cleavage_records=(cleavage_record,),
                        )
                        add_edges(edge)

                next_node_ids = list(new_node_ids)
                print(f"Depth {depth} completed. New nodes: {len(new_node_ids)}")
                print(f"Total nodes so far: {len(nodes)}")
                print(f"Total edges so far: {len(edges)}")
                print()
        
        fragment_tree = FragmentTree(compound=compound, nodes=nodes, edges=edges)
        return fragment_tree
                
    def copy(self) -> 'Fragmenter':
        """
        Create a copy of the current Fragmenter instance.

        Returns:
            Fragmenter: A new instance with the same adduct_types and max_depth.
        """
        return Fragmenter(
            adduct_type=tuple(self.adduct_type),
            max_depth=self.max_depth,
            cleavage_patterns=tuple(self.patterns)
        )


@dataclass(frozen=True)
class FragmentInfo:
    smiles: str
    adduct: Adduct
    parent_smiles: str
    bond_positions: Tuple[int]
    cleavage_pattern: Tuple[str, int]  # (smirks, fragment_index)

    def __hash__(self):
        return hash((self.adduct, self.parent_smiles, self.bond_positions, self.cleavage_pattern))

    def __eq__(self, other):
        if not isinstance(other, FragmentInfo):
            return False
        return (self.adduct == other.adduct and
                self.parent_smiles == other.parent_smiles and
                self.bond_positions == other.bond_positions and
                self.cleavage_pattern == other.cleavage_pattern)