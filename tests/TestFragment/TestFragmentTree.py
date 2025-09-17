import unittest
from rdkit import Chem
from MassMolKit.Fragment.CleavagePattern import CleavagePattern
from MassMolKit.Fragment.cleavage_patterns import patterns as default_cleavage_patterns
from MassMolKit.Mol.Compound import Compound
from MassMolKit.Mol.Formula import Formula
from MassMolKit.Fragment.Fragmenter import Fragmenter
from MassMolKit.MS.constants import AdductType


class TestFragmentTree(unittest.TestCase):

    def setUp(self):
        """Prepare common test molecules"""
        self.patterns = default_cleavage_patterns
        self.acetamide = Compound(Chem.MolFromSmiles("CC(=O)NC"))      # Acetamide
        self.ethyl_acetate = Compound(Chem.MolFromSmiles("CC(=O)OCC")) # Ethyl acetate
        self.trimethylammonium = Compound(Chem.MolFromSmiles("C[N+](C)(C)C")) # Trimethylammonium
        self.deamino_nTrp_DL_Ala_OH = Compound(Chem.MolFromSmiles("CC(NC(=O)CC1=CNC2=C1C=CC=C2)C(O)=O"))
        self.chlorobenzene = Compound.from_smiles("c1ccccc1Cl") # Chlorobenzene

    def test_tree_construction(self):
        fragmenter = Fragmenter(
            adduct_type=(AdductType.M_PLUS_H_POS,), 
            max_depth=8,
            cleavage_patterns=self.patterns,
            )
        fragment_tree = fragmenter.create_fragment_tree(self.deamino_nTrp_DL_Ala_OH)
        all_formulas = fragment_tree.get_all_formulas(sources=True)
        # fragment_tree.save_topological_tsv("./node.tsv", "./edge.tsv")
        pass

    def test_chlorobenzene_tree(self):
        fragmenter = Fragmenter(
            adduct_type=(AdductType.M_PLUS_H_POS,), 
            max_depth=8,
            cleavage_patterns=self.patterns,
            )
        fragment_tree = fragmenter.create_fragment_tree(self.chlorobenzene)
        all_formulas = fragment_tree.get_all_formulas(sources=True)
        # fragment_tree.save_topological_tsv("./node.tsv", "./edge.tsv")
        pass
