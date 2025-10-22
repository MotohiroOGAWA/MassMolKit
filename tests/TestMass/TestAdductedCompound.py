import unittest
from mmkit.chem.Formula import Formula
from mmkit.chem.Compound import Compound
from mmkit.mass.Adduct import Adduct
from mmkit.mass.AdductedCompound import AdductedCompound


class TestAdductedCompound(unittest.TestCase):
    def setUp(self):
        # Ethanol as base compound
        self.ethanol = Compound.from_smiles("CCO")  # formula: C2H6O
        self.h = Formula.parse("H+")             # proton
        self.na = Formula.parse("Na+")           # sodium ion

    def test_formula_with_proton(self):
        # Ethanol + H+
        adduct = Adduct(ion_type="M", adducts_in=[self.h])
        ion = AdductedCompound(self.ethanol, adduct)
        f = ion.formula
        self.assertIn("H", f.value)
        self.assertEqual(f.charge, 1,
                         msg="Charge mismatch in formula with proton adduct")

    def test_charge_property(self):
        # Ethanol + Na+
        adduct = Adduct(ion_type="M", adducts_in=[self.na])
        ion = AdductedCompound(self.ethanol, adduct)
        self.assertEqual(ion.charge, 1,
                         msg="Charge property mismatch for sodium adduct")

    def test_mz_calculation(self):
        # Ethanol + H+ (should give m/z ~ 47.049)
        adduct = Adduct(ion_type="M", adducts_in=[self.h])
        ion = AdductedCompound(self.ethanol, adduct)
        mz_val = ion.mz
        expected_mass = ion.formula.exact_mass
        self.assertAlmostEqual(mz_val, expected_mass / ion.charge, places=6)

    def test_zero_charge_raises(self):
        # Artificially construct adduct with 0 charge
        adduct = Adduct(ion_type="M", adducts_in=[], adducts_out=[])
        ion = AdductedCompound(self.ethanol, adduct)
        ion.adduct._charge = 0  # force 0 charge
        with self.assertRaises(ValueError):
            _ = ion.mz

    def test_repr(self):
        adduct = Adduct(ion_type="M", adducts_in=[self.h])
        ion = AdductedCompound(self.ethanol, adduct)
        rep = repr(ion)
        self.assertIn("AdductedCompound", rep)
        self.assertIn("CCO", rep)  # ethanol smiles
        self.assertIn("[M+H]+", rep)  # adduct string

    def test_str_and_parse_with_proton(self):
        # Ethanol + H+
        adduct = Adduct(ion_type="M", adducts_in=[self.h])
        ion = AdductedCompound(self.ethanol, adduct)

        # Convert to string
        s = str(ion)
        self.assertIn("[M+H]+", s)
        self.assertIn("CCO", s)

        # Parse back
        parsed = AdductedCompound.parse(s)
        self.assertEqual(parsed.compound.formula, ion.compound.formula)
        self.assertEqual(parsed.adduct.charge, ion.adduct.charge)

    def test_str_and_parse_with_sodium(self):
        # Ethanol + Na+
        adduct = Adduct(ion_type="M", adducts_in=[self.na])
        ion = AdductedCompound(self.ethanol, adduct)

        s = str(ion)
        self.assertIn("[M+Na]+", s)

        parsed = AdductedCompound.parse(s)
        self.assertEqual(parsed.charge, ion.charge)
        self.assertAlmostEqual(parsed.mz, ion.mz, places=6)

    def test_parse_invalid_string(self):
        # Missing delimiter
        with self.assertRaises(ValueError):
            AdductedCompound.parse("C2H6O[M+H]+".replace(AdductedCompound.DELIM, ""))


if __name__ == "__main__":
    unittest.main()
