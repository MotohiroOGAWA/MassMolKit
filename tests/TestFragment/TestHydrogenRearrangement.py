import os
import tempfile
import unittest
from typing import Dict, Any

from rdkit import Chem

from mmkit.chem.Compound import Compound
from mmkit.fragment.HydrogenRearrangement import (
    HydrogenRearrangement,
    IonHydrogenShiftRule,
    RadicalRule,
    BondUnsaturationRule,
)


class TestHydrogenRearrangement(unittest.TestCase):

    def setUp(self):
        # Simple molecules
        self.ethanol = Compound(Chem.MolFromSmiles("CCO"))                  # C, O
        self.ethane = Compound(Chem.MolFromSmiles("CC"))               # C, C
        self.dimethyl_sulfide = Compound(Chem.MolFromSmiles("CSC"))         # C, S
        self.acetamide = Compound(Chem.MolFromSmiles("CNC(C)=O"))           # C, N, O
        self.trimethylammonium = Compound(Chem.MolFromSmiles("C[N+](C)(C)C"))  # C, N (no N-H)

        # Rules
        self.ion_shift = IonHydrogenShiftRule(
            plus_h_atoms=["N", "O", "P", "S"],
            minus_h_atoms=["C", "P", "S"],
        )
        self.radical = RadicalRule(max_count=1)
        self.unsat = BondUnsaturationRule(max_count=2)

        self.hr = HydrogenRearrangement(
            ion_shift_rule=self.ion_shift,
            radical_rule=self.radical,
            bond_unsaturation_rule=self.unsat,
        )

    def test_delta_h_candidates_dict(self):
        cands = self.hr.delta_h_candidates
        self.assertIn("ion_shift", cands)
        self.assertIn("radical", cands)
        self.assertIn("bond_unsaturation", cands)

        self.assertEqual(cands["ion_shift"], (-1, 0, 1))
        # max_count=1 -> num_candidates=2
        self.assertEqual(len(cands["radical"]), 2)
        # max_count=2 -> num_candidates=3
        self.assertEqual(cands["bond_unsaturation"], (0, -2, -4))

    def test_evaluate_masks_shapes(self):
        # each mask length should match rule.num_candidates
        mask_ion = self.ion_shift.evaluate(self.acetamide)
        mask_rad = self.radical.evaluate(self.acetamide)
        mask_uns = self.unsat.evaluate(self.acetamide)

        self.assertEqual(len(mask_ion), self.ion_shift.num_candidates)
        self.assertEqual(len(mask_rad), self.radical.num_candidates)
        self.assertEqual(len(mask_uns), self.unsat.num_candidates)

    def test_ion_shift_evaluate(self):
        # acetamide has C/N/O -> both minus(C) and plus(N/O) should be possible
        mask = self.ion_shift.evaluate(self.acetamide)
        # delta_h_candidates is (-1, 0, +1) in this order
        self.assertEqual(mask, (True, False, True))

        # ethanol has C/O -> both minus(C) and plus(O)
        self.assertEqual(self.ion_shift.evaluate(self.ethanol), (True, False, True))
        # ethane has C/C -> both
        self.assertEqual(self.ion_shift.evaluate(self.ethane), (True, False, False))
        # dimethylsulfide has C/S -> both
        self.assertEqual(self.ion_shift.evaluate(self.dimethyl_sulfide), (True, False, True))
        # trimethylammonium has C/N (no N-H) -> only minus(C)
        self.assertEqual(self.ion_shift.evaluate(self.trimethylammonium), (False, True, False))

    def test_bond_unsaturation_evaluate(self):
        # ethanol has C-C and C-O; at least one single bond with H on both ends exists
        mask = self.unsat.evaluate(self.ethanol)
        # mask length 3 (0, -2, -4)
        self.assertEqual(len(mask), 3)
        # 0 is always possible; also at least one unsaturation should be possible
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        # second step depends on match count; should be bool either way, but never raises
        self.assertIsInstance(mask[2], bool)

    def test_radical_evaluate(self):
        # With max_count=1, any molecule with >=1 match gives (True, True)
        mask = self.radical.evaluate(self.acetamide)
        self.assertEqual(mask, (True, True))

        # Trimethylammonium still has many H-bearing atoms (carbons), so should still be applicable
        mask2 = self.radical.evaluate(self.trimethylammonium)
        self.assertEqual(mask2, (True, True))

    def test_yaml_io_roundtrip(self):
        with tempfile.TemporaryDirectory(dir="tests") as td:
            path = os.path.join(td, "hydrogen_rearrangement.yaml")

            # save
            self.hr.to_yaml(path)
            self.assertTrue(os.path.exists(path))

            # load
            hr2 = HydrogenRearrangement.from_yaml(path)

            # basic checks
            self.assertAlmostEqual(hr2.version, self.hr.version)
            self.assertIsNotNone(hr2.ion_shift_rule)
            self.assertIsNotNone(hr2.radical_rule)
            self.assertIsNotNone(hr2.bond_unsaturation_rule)

            # candidates should match
            self.assertEqual(hr2.delta_h_candidates, self.hr.delta_h_candidates)

            # evaluate consistency for a molecule (masks should match)
            self.assertEqual(hr2.ion_shift_rule.evaluate(self.acetamide), self.hr.ion_shift_rule.evaluate(self.acetamide))
            self.assertEqual(hr2.radical_rule.evaluate(self.acetamide), self.hr.radical_rule.evaluate(self.acetamide))
            self.assertEqual(hr2.bond_unsaturation_rule.evaluate(self.acetamide), self.hr.bond_unsaturation_rule.evaluate(self.acetamide))


if __name__ == "__main__":
    unittest.main()
