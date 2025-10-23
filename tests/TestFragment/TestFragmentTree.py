import unittest
import os
import re
import glob
from typing import List, Tuple, Dict
from rdkit import Chem
from mmkit.fragment.CleavagePattern import CleavagePattern
from mmkit.fragment.CleavagePatternLibrary import CleavagePatternLibrary
from mmkit.chem.Compound import Compound
from mmkit.chem.formula_utils import assign_formulas_to_peaks
from mmkit.mass.Adduct import Adduct
from mmkit.chem.Formula import Formula
from mmkit.fragment.Fragmenter import Fragmenter
from mmkit.mass.constants import AdductType, IonMode, parse_ion_mode
from mmkit.fragment.FragmentTree import FragmentTree, FragmentNode, FragmentEdge
from mmkit.fragment.AdductedFragmentTree import AdductedFragmentTree
from mmkit.mass.Tolerance import PpmTolerance, DaTolerance
from mmkit.fragment.FragmentPathway import *


class TestFragmentTree(unittest.TestCase):
    def setUp(self):
        """Prepare common test molecules"""
        self.temp_files = []  # Register the temporary file for cleanup

        self.cleavage_pattern_lib = CleavagePatternLibrary.load_default_positive()
        dir = os.path.join('tests', 'dummy_files', 'test_fragment_tree')
        pattern = re.compile(r"^compound\d+.*\.txt$", re.IGNORECASE)
        self.input_files = [
            os.path.join(dir, f) for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and pattern.match(f)
        ]
        
        self.acetamide = Compound(Chem.MolFromSmiles("CC(=O)NC"))      # Acetamide
        self.ethyl_acetate = Compound(Chem.MolFromSmiles("CC(=O)OCC")) # Ethyl acetate
        self.trimethylammonium = Compound(Chem.MolFromSmiles("C[N+](C)(C)C")) # Trimethylammonium
        self.deamino_nTrp_DL_Ala_OH = Compound(Chem.MolFromSmiles("CC(NC(=O)CC1=CNC2=C1C=CC=C2)C(O)=O"))
        self.chlorobenzene = Compound.from_smiles("c1ccccc1Cl") # Chlorobenzene

    def tearDown(self):
        """Clean up any temporary files created during testing."""
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

    # ----------------------------------------------------------------------
    def test_fragmenter_serialization(self):
        """Test that Fragmenter can be correctly saved to and loaded from JSON."""
        fragmenter = Fragmenter(
            max_depth=3,
            cleavage_pattern_lib=self.cleavage_pattern_lib
        )

        temp_path = os.path.join(
            'tests', 'dummy_files', 'test_fragment_tree', 'temp', 'temp_fragmenter.json'
        )
        self.temp_files.append(temp_path)

        # Save and load
        fragmenter.save_json(temp_path)
        loaded_fragmenter = Fragmenter.load_json(temp_path)

        # Verify equality
        self.assertEqual(fragmenter.max_depth, loaded_fragmenter.max_depth)
        self.assertEqual(
            len(fragmenter.cleavage_pattern_lib),
            len(loaded_fragmenter.cleavage_pattern_lib)
        )

    def test_tree_construction(self):
        for input_file in self.input_files:
            smiles = None
            adduct_str = None
            ion_mode_str = None
            peaks:list[float, float] = []
            with open(input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SMILES:"):
                        smiles = line.split(":", 1)[1].strip()
                    elif line.startswith("PRECURSORTYPE:") or line.startswith("AdductType:"):
                        adduct_str = line.split(":", 1)[1].strip()
                    elif line.startswith("IONMODE:") or line.startswith("IonMode:"):
                        ion_mode_str = line.split(":", 1)[1].strip()
                    elif re.match(r"^\d", line):
                        parts = line.split()
                        if len(parts) >= 2:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            peaks.append((mz, intensity))
            fragmenter = Fragmenter(
                max_depth=2,
                cleavage_pattern_lib=self.cleavage_pattern_lib
                )
            if smiles is None or adduct_str is None:
                continue
            
            compound = Compound.from_smiles(smiles)
            adduct = Adduct.parse(adduct_str)
            ion_mode = parse_ion_mode(ion_mode_str)
            fragment_tree = fragmenter.create_fragment_tree(compound, ion_mode=ion_mode)
            
            node_str = fragment_tree.nodes[1].__str__()
            node = FragmentNode.parse(node_str)
            
            edge_str = fragment_tree.edges[(0,2)].__str__()
            edge = FragmentEdge.parse(edge_str)
            fragment_step = FragmentStep.parse(edge.fragment_step_strs[0])

            adducted_tree = AdductedFragmentTree(fragment_tree)

            fragment_pathways: List[FragmentPathway] = []
            fragment_pathways = FragmentPathway.build_pathways_by_peak(
                adducted_tree=adducted_tree,
                adduct_type=AdductType.M_PLUS_H_POS,
                peaks_mz=[p[0] for p in peaks],
                mass_tolerance=PpmTolerance(10),
            )
            fragment_pathway_peak4_0 = fragment_pathways[4][0]
            fragment_pathway_peak4_0_str = str(fragment_pathway_peak4_0)
            fragment_pathway_peak4_0_parsed = FragmentPathway.parse(fragment_pathway_peak4_0_str)
            pass
