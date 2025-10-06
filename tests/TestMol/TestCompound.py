import unittest
from rdkit import Chem
from mmkit.chem.Compound import Compound
from mmkit.chem.Formula import Formula


class TestCompound(unittest.TestCase):
    def setUp(self):
        # Ethanol as test molecule
        self.ethanol = Compound.from_smiles("CCO")  # canonical SMILES: CCO
        self.na_cation = Compound.from_smiles("[Na+]")

    def test_smiles_and_repr(self):
        # Canonical SMILES should be returned
        self.assertEqual(self.ethanol.smiles, "CCO")
        self.assertIn("Compound(smiles=CCO)", repr(self.ethanol))
        self.assertEqual(str(self.ethanol), "CCO")

    def test_formula_and_mass(self):
        # Ethanol should have formula C2H6O
        self.assertEqual(self.ethanol.formula.value, "C2H6O")
        # exact_mass should be consistent with Formula
        expected_mass = Formula.from_mol(self.ethanol.mol).exact_mass
        self.assertAlmostEqual(self.ethanol.exact_mass, expected_mass, places=4)
        # Na+ should have +1 charge
        self.assertEqual(self.na_cation.charge, 1)

    def test_copy(self):
        eth_copy = self.ethanol.copy()
        self.assertEqual(self.ethanol.smiles, eth_copy.smiles)
        self.assertIsNot(self.ethanol, eth_copy)

    def test_with_atom_map_and_get_index(self):
        # Create with atom map numbers
        mapped = self.ethanol.with_atom_map(inplace=False, overwrite=True)
        # All atoms should have atom map numbers
        self.assertTrue(all(atom.GetAtomMapNum() > 0 for atom in mapped._mol.GetAtoms()))
        # Get atom index from map number should return a valid index
        idx = mapped.get_atom_index_from_map(1)
        self.assertIsInstance(idx, int)

    def test_atom_map_to_idx(self):
        mapped = self.ethanol.with_atom_map(inplace=False, overwrite=True)
        amap = mapped.atom_map_to_idx
        # Should be a bidict with length equal to number of atoms
        self.assertEqual(len(amap), mapped._mol.GetNumAtoms())
        # Check that reverse lookup works
        first_key = list(amap.keys())[0]
        first_val = amap[first_key]
        self.assertEqual(amap.inv[first_val], first_key)


if __name__ == "__main__":
    unittest.main()
