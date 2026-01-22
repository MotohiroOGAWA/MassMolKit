import unittest
from rdkit import Chem
from mmkit.fragment.CleavagePattern import CleavagePattern
from mmkit.fragment.CleavagePatternSet import CleavagePatternSet
from mmkit.chem.Compound import Compound
from mmkit.chem.Formula import Formula


class TestCleavagePattern(unittest.TestCase):

    def setUp(self):
        """Prepare common test molecules"""
        self.cleavage_pattern_set = CleavagePatternSet.load_default_positive()
        self.acetamide = Compound(Chem.MolFromSmiles("CNC(C)=O"))      # Acetamide
        self.ethyl_acetate = Compound(Chem.MolFromSmiles("CC(=O)OCC")) # Ethyl acetate
        self.trimethylammonium = Compound(Chem.MolFromSmiles("C[N+](C)(C)C")) # Trimethylammonium

    def test_cleavage_pattern_application(self):
        """Test cleavage pattern application on molecules"""
        result = self.cleavage_pattern_set.patterns[0].fragment(self.acetamide)
        result0 = self.cleavage_pattern_set.patterns[0].fragment_at_atom_idx(self.acetamide, (0, 1))
        result1 = self.cleavage_pattern_set.patterns[0].fragment_at_atom_idx(self.acetamide, (1, 0))
        result2 = self.cleavage_pattern_set.patterns[0].fragment_at_atom_idx(self.acetamide, (2, 3))
        result3 = self.cleavage_pattern_set.patterns[0].fragment_at_atom_idx(self.acetamide, (3, 2))
        result4 = self.cleavage_pattern_set.patterns[0].fragment_at_atom_idx(self.acetamide, (4, 5))
        pass

    