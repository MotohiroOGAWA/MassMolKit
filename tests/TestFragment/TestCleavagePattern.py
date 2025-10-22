import unittest
from rdkit import Chem
from mmkit.fragment.CleavagePattern import CleavagePattern
from mmkit.fragment.CleavagePatternLibrary import CleavagePatternLibrary
from mmkit.chem.Compound import Compound
from mmkit.chem.Formula import Formula


class TestCleavagePattern(unittest.TestCase):

    def setUp(self):
        """Prepare common test molecules"""
        self.cleavage_pattern_lib = CleavagePatternLibrary.load_default_positive()
        self.acetamide = Compound(Chem.MolFromSmiles("CC(=O)NC"))      # Acetamide
        self.ethyl_acetate = Compound(Chem.MolFromSmiles("CC(=O)OCC")) # Ethyl acetate
        self.trimethylammonium = Compound(Chem.MolFromSmiles("C[N+](C)(C)C")) # Trimethylammonium

    