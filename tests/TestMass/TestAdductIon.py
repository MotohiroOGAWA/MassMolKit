import unittest
from MassMolKit.Mol.Formula import Formula
from MassMolKit.Mol.Compound import Compound
from MassMolKit.MS.Adduct import Adduct
from MassMolKit.MS.AdductIon import AdductIon


class TestAdductIon(unittest.TestCase):
    def setUp(self):
        # Ethanol as base compound
        self.ethanol = Compound.from_smiles("CCO")  # formula: C2H6O
        self.h = Formula.from_str("H+")             # proton
        self.na = Formula.from_str("Na+")           # sodium ion

    def test_formula_with_proton(self):
        # Ethanol + H+
        adduct = Adduct(mode="M", adducts_in=[self.h])
        ion = AdductIon(self.ethanol, adduct)
        f = ion.formula
        self.assertIn("H", f.value)
        self.assertEqual(f.charge, 1,
                         msg="Charge mismatch in formula with proton adduct")

    def test_charge_property(self):
        # Ethanol + Na+
        adduct = Adduct(mode="M", adducts_in=[self.na])
        ion = AdductIon(self.ethanol, adduct)
        self.assertEqual(ion.charge, 1,
                         msg="Charge property mismatch for sodium adduct")

    def test_mz_calculation(self):
        # Ethanol + H+ (should give m/z ~ 47.049)
        adduct = Adduct(mode="M", adducts_in=[self.h])
        ion = AdductIon(self.ethanol, adduct)
        mz_val = ion.mz
        expected_mass = ion.formula.exact_mass
        self.assertAlmostEqual(mz_val, expected_mass / ion.charge, places=6)

    def test_zero_charge_raises(self):
        # Artificially construct adduct with 0 charge
        adduct = Adduct(mode="M", adducts_in=[], adducts_out=[])
        ion = AdductIon(self.ethanol, adduct)
        ion.adduct._charge = 0  # force 0 charge
        with self.assertRaises(ValueError):
            _ = ion.mz

    def test_repr(self):
        adduct = Adduct(mode="M", adducts_in=[self.h])
        ion = AdductIon(self.ethanol, adduct)
        rep = repr(ion)
        self.assertIn("AdductedIon", rep)
        self.assertIn("CCO", rep)  # ethanol smiles
        self.assertIn("[M+H]+", rep)  # adduct string


if __name__ == "__main__":
    unittest.main()
