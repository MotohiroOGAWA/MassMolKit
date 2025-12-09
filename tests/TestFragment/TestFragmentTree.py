import unittest
import os
import re
import glob
from typing import List, Tuple, Dict
from rdkit import Chem
from mmkit.fragment.CleavagePatternLibrary import CleavagePatternLibrary
from mmkit.chem.Compound import Compound
from mmkit.chem.formula_utils import assign_formulas_to_peaks
from mmkit.mass.Adduct import Adduct
from mmkit.fragment.Fragmenter import Fragmenter
from mmkit.mass.constants import parse_ion_mode
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
        

    # ----------------------------------------------------------------------
    def test_fragmenter_serialization(self):
        """Test that Fragmenter can be correctly saved to and loaded from JSON."""
        fragmenter = Fragmenter(
            max_depth=3,
            adduct_types=self.supported_adduct_types,
            cleavage_pattern_lib=self.cleavage_pattern_lib
        )

        temp_path = os.path.join(
            'tests', 'dummy_files', 'test_fragment_tree', 'temp', 'temp_fragmenter.yaml'
        )
        self.temp_files.append(temp_path)

        # Save and load
        fragmenter.save_yaml(temp_path)
        loaded_fragmenter = Fragmenter.load_yaml(temp_path)

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

            if smiles is None or adduct_str is None or ion_mode_str is None:
                continue  # Skip invalid files
            compound = Compound.from_smiles(smiles)
            precursor_type = Adduct.parse(adduct_str)
            ion_mode = parse_ion_mode(ion_mode_str)
            
            # --- Fragmenter ---
            fragmenter = Fragmenter(
                max_depth=8,
                adduct_types=self.supported_adduct_types,
                cleavage_pattern_lib=self.cleavage_pattern_lib,
            )

            # --- Fragment tree ---
            fragment_tree = fragmenter.create_fragment_tree(
                compound,
                ion_mode=ion_mode,
                timeout_seconds=5,
            )
            self.assertIsNotNone(fragment_tree)
            depths = fragment_tree.get_nodes_by_depth()

            # --- Precursor pathways ---
            precursor_pathways = fragmenter.build_fragment_pathways_for_precursor(
                fragment_tree=fragment_tree,
                precursor_type=precursor_type,
            )
            self.assertTrue(len(precursor_pathways) > 0, "Precursor pathways should not be empty")

            # Convert to string and parse back
            precursor_pathway_str = fragmenter.list_to_str(precursor_pathways)
            precursor_parsed = fragmenter.parse_list(precursor_pathway_str)
            self.assertEqual(len(precursor_pathways), len(precursor_parsed))

            # --- Peak-level pathways ---
            peaks_mz = [mz for mz, intensity in peaks]
            fragment_pathways_by_peak = fragmenter.build_fragment_pathways_by_peak(
                fragment_tree=fragment_tree,
                precursor_type=precursor_type,
                peaks_mz=peaks_mz,
                mass_tolerance=DaTolerance(0.01),
            )
            self.assertEqual(len(fragment_pathways_by_peak), len(peaks_mz))

            # For each peak, validate that parsing works
            for fp_list in fragment_pathways_by_peak:
                if len(fp_list) == 0:
                    continue

                fp_str = fragmenter.list_to_str(fp_list)
                fp_parsed = fragmenter.parse_list(fp_str)
                self.assertEqual(len(fp_list), len(fp_parsed))

            # If all passes
            print("✓ precursor pathway test passed")
