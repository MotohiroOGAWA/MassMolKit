import unittest
from rdkit import Chem
from MassMolKit.Fragment.CleavagePattern import CleavagePattern
from MassMolKit.Fragment.cleavage_patterns import patterns as default_cleavage_patterns
from MassMolKit.Mol.Compound import Compound
from MassMolKit.Mol.Formula import Formula


class TestCleavagePattern(unittest.TestCase):

    def setUp(self):
        """Prepare common test molecules"""
        self.patterns = default_cleavage_patterns
        self.acetamide = Compound(Chem.MolFromSmiles("CC(=O)NC"))      # Acetamide
        self.ethyl_acetate = Compound(Chem.MolFromSmiles("CC(=O)OCC")) # Ethyl acetate
        self.trimethylammonium = Compound(Chem.MolFromSmiles("C[N+](C)(C)C")) # Trimethylammonium

    