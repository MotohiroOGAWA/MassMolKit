import unittest
import os
import re
import glob
import numpy as np
from typing import List, Tuple, Dict
from rdkit import Chem
from mmkit.fragment.CleavagePatternSet import CleavagePatternSet
from mmkit.fragment.HydrogenRearrangement import HydrogenRearrangement
from mmkit.chem.Compound import Compound
from mmkit.chem.formula_utils import assign_formulas_to_peaks
from mmkit.mass.Adduct import Adduct
from mmkit.fragment.FragmentTreeBuilder import FragmentTreeBuilder
from mmkit.fragment.Fragmenter import Fragmenter
from mmkit.mass.constants import parse_ion_mode
from mmkit.mass.Tolerance import PpmTolerance, DaTolerance, AnyDaPpmTolerance
from mmkit.fragment.FragmentPathway import *


class TestFragmentTree(unittest.TestCase):
    def setUp(self):
        """Prepare common test molecules"""
        self.temp_files = []  # Register the temporary file for cleanup

        self.cleavage_pattern_set = CleavagePatternSet.load_default_positive()
        self.hydrogen_rearrangement = HydrogenRearrangement.load_default_positive()
        dir = os.path.join('tests', 'dummy_files', 'test_fragment_tree')
        pattern = re.compile(r"^compound\d+.*\.txt$", re.IGNORECASE)
        self.input_files = [
            os.path.join(dir, f) for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and pattern.match(f)
        ]

        self.supported_adduct_types = (Adduct.parse("[M+H]+"), Adduct.parse("[M+NH4]+"), Adduct.parse("[M+Na]+"))
        
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

    def test_cleavage_patterns(self):
        cleavage = CleavagePattern(smirks='[#6,#7,#8:1]-[#6,#7,#8:2]-[#9,#17,#35,#53:3]>>[*:1]=[*:2]',charge_mode='positive1')
        c = Compound.from_smiles("BrC(Br)(Br)[C+]c1cnc2ccccc2n1")  # Cycloheptane
        result = cleavage.fragment(c)  # Precompute the cleavage
        pass

        cleavage = CleavagePattern(smirks='[*:1]-[*:2]1-[*:3]-[*:4]-[*:5]-[*:6]-[*:7]-1>>[*:1]-[*:2]=[*:7]',charge_mode='positive1')
        c = Compound.from_smiles("[C+]CC1CCCCC1")  # Cycloheptane
        result = cleavage.fragment(c)  # Precompute the cleavage
        
    def test_fragmenter_serialization(self):
        """Test that Fragmenter can be correctly saved to and loaded from JSON."""
        fragment_tree_builder = FragmentTreeBuilder(
            max_depth=3,
            cleavage_pattern_set=self.cleavage_pattern_set,
        )
        fragmenter = Fragmenter(
            ion_mode=parse_ion_mode("positive"),
            fragment_tree_builder=fragment_tree_builder,
            adduct_types=self.supported_adduct_types,
            hydrogen_rearrangement=self.hydrogen_rearrangement,
        )

        temp_path = os.path.join(
            'tests', 'dummy_files', 'test_fragment_tree', 'temp', 'temp_fragmenter.yaml'
        )
        self.temp_files.append(temp_path)

        # Save and load
        fragmenter.to_yaml(temp_path)
        loaded_fragmenter = Fragmenter.from_yaml(temp_path)

        # Verify equality
        self.assertEqual(fragmenter.tree_max_depth, loaded_fragmenter.tree_max_depth)
        self.assertEqual(
            len(fragmenter.cleavage_pattern_set),
            len(loaded_fragmenter.cleavage_pattern_set)
        )

    def test_fragment_tree_serialization(self):
        """Test that FragmentTree can be correctly saved to and loaded from JSON."""
        fragment_tree_builder = FragmentTreeBuilder(
            max_depth=3,
            cleavage_pattern_set=self.cleavage_pattern_set,
        )
        compound = self.acetamide

        fragment_tree = fragment_tree_builder.create_fragment_tree(
            compound,
            timeout_seconds=float('inf'),
        )
        depths = fragment_tree.get_nodes_by_depth()
        self.assertEqual(fragment_tree.smiles, compound.smiles)

        # ---- Save to HDF5 (overwrite) ----
        tmp_path = os.path.join("tests", "dummy_files", "test_fragment_tree", "tmp_fragment_tree.h5")
        self.temp_files.append(tmp_path)

        tree_id = "acetamide_pos_depth3"
        fragment_tree.to_hdf5(tmp_path, tree_id=tree_id, mode="w")

        # ---- Ensure tree_id is recorded in metadata ----
        stored_ids = FragmentTree.list_tree_ids(tmp_path)
        self.assertIn(tree_id, stored_ids)

        # ---- Load back ----
        loaded_tree = FragmentTree.from_hdf5(tmp_path, tree_id=tree_id)

        # ---- Basic equality checks ----
        self.assertEqual(fragment_tree.smiles, loaded_tree.smiles)
        self.assertEqual(fragment_tree.num_nodes, loaded_tree.num_nodes)
        self.assertEqual(fragment_tree.num_edges, loaded_tree.num_edges)

        # ---- Array equality checks ----
        np.testing.assert_array_equal(fragment_tree._node_smiles, loaded_tree._node_smiles)
        np.testing.assert_array_equal(fragment_tree._edge_index, loaded_tree._edge_index)
        np.testing.assert_array_equal(fragment_tree._edge_step_indptr, loaded_tree._edge_step_indptr)
        np.testing.assert_array_equal(fragment_tree._edge_step_flat, loaded_tree._edge_step_flat)

        # ---- Optional CSR caches: build in both, then compare ----
        fragment_tree._build_edge_adjacency()
        loaded_tree._build_edge_adjacency()
        np.testing.assert_array_equal(fragment_tree._in_edge_ids, loaded_tree._in_edge_ids)
        np.testing.assert_array_equal(fragment_tree._in_edge_indptr, loaded_tree._in_edge_indptr)
        np.testing.assert_array_equal(fragment_tree._out_edge_ids, loaded_tree._out_edge_ids)
        np.testing.assert_array_equal(fragment_tree._out_edge_indptr, loaded_tree._out_edge_indptr)
        # ---- Check overwrite semantics (same tree_id should overwrite) ----
        # Create a different tree (different compound) and overwrite same tree_id in append mode
        fragment_tree2 = fragment_tree_builder.create_fragment_tree(
            self.ethyl_acetate,
            timeout_seconds=float('inf'),
        )
        fragment_tree2.to_hdf5(tmp_path, tree_id=tree_id, mode="a")  # overwrite expected

        loaded_tree2 = FragmentTree.from_hdf5(tmp_path, tree_id=tree_id)

        # Now it should match fragment_tree2, not fragment_tree
        self.assertEqual(fragment_tree2.smiles, loaded_tree2.smiles)
        self.assertEqual(fragment_tree2.num_nodes, loaded_tree2.num_nodes)
        self.assertEqual(fragment_tree2.num_edges, loaded_tree2.num_edges)
        np.testing.assert_array_equal(fragment_tree2._edge_index, loaded_tree2._edge_index)

        # ---- Append a second tree_id and confirm metadata updated ----
        tree_id2 = "ethyl_acetate_pos_depth3"
        fragment_tree2.to_hdf5(tmp_path, tree_id=tree_id2, mode="a")

        stored_ids2 = FragmentTree.list_tree_ids(tmp_path)
        self.assertIn(tree_id, stored_ids2)
        self.assertIn(tree_id2, stored_ids2)

    def test_pathway_construction(self):
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

            if smiles is None or adduct_str is None or ion_mode_str is None:
                continue  # Skip invalid files
            compound = Compound.from_smiles(smiles)
            precursor_type = Adduct.parse(adduct_str)
            ion_mode = parse_ion_mode(ion_mode_str)
            
            # --- Fragmenter ---
            fragment_tree_builder = FragmentTreeBuilder(
                max_depth=3,
                cleavage_pattern_set=self.cleavage_pattern_set,
                only_add_min_depth=False,
            )
            fragmenter = Fragmenter(
                ion_mode=parse_ion_mode("positive"),
                fragment_tree_builder=fragment_tree_builder,
                adduct_types=self.supported_adduct_types,
                hydrogen_rearrangement=self.hydrogen_rearrangement,
            )

            # --- Fragment tree ---
            h_fragment_tree = fragmenter.create_hydrogen_rearranged_fragment_tree(
                compound,
                timeout_seconds=float('inf'),
            )
            self.assertIsNotNone(h_fragment_tree)
            self.assertEqual(h_fragment_tree.smiles, compound.smiles)

            # --- Peak-level pathways ---
            mass_tolerance = AnyDaPpmTolerance(mode="ppm", da_tolerance=0.01, ppm_tolerance=10.0)
            peaks_mz = [mz for mz, intensity in peaks]
            peaks_intensity = [intensity for mz, intensity in peaks]
            precursor_fragment_pathways, fragment_pathways_by_peak = fragmenter.build_fragment_pathways_by_peak(
                h_fragment_tree=h_fragment_tree,
                precursor_type=precursor_type,
                peaks_mz=peaks_mz,
                mass_tolerance=mass_tolerance,
            )
            self.assertEqual(len(fragment_pathways_by_peak), len(peaks_mz))

            pfp_group_str = str(precursor_fragment_pathways)
            pfp_parsed = fragmenter.parse_fragment_pathway_group(pfp_group_str)
            for pw in pfp_parsed.pathways:
                has_precursor = any(node.is_precursor for node in pw.nodes)
                self.assertTrue(has_precursor, msg=f"Pathway {pw} does not contain a precursor node")

            # For each peak, validate that parsing works
            sum_intensity = 0.0
            matched_intensity = 0.0
            for fp_group, intensity in zip(fragment_pathways_by_peak, peaks_intensity):
                sum_intensity += intensity
                if len(fp_group) == 0:
                    continue
                matched_intensity += intensity

                fp_group_str = str(fp_group)
                if len(fp_group[0].path) > 1:
                    FragmentPathwayEdge.parse(str(fp_group[0].get_edge(0)))
                fp_parsed = fragmenter.parse_fragment_pathway_group(fp_group_str)
                self.assertEqual(len(fp_group), len(fp_parsed))
                # if len(fp_parsed.with_precursor.shortest) == 0:
                #     print(f"Failed to find precursor in parsed fragment pathways for peak with m/z {peaks_mz[0]} and intensity {intensity}: {fp_parsed}")

                # self.assertGreater(len(fp_parsed.with_precursor.shortest), 0, msg=f"Failed to find precursor in parsed fragment pathways: {fp_parsed}")

            coverage = matched_intensity / sum_intensity if sum_intensity > 0 else 0.0
            pass