import unittest
from mmkit.chem.Formula import Formula
from mmkit.mass.Adduct import Adduct


class TestAdduct(unittest.TestCase):
    def setUp(self):
        # Prepare basic formulas for testing
        self.h = Formula.parse("H+")  # proton
        self.na = Formula.parse("Na+")     # sodium ion
        self.h2o = Formula.parse("H2O")   # neutral molecule

    def test_format(self):
        # Test string representation
        adduct = Adduct.parse("[M+H]+")
        adduct = Adduct.parse("[M+H-H2O]+")
        adduct = Adduct.parse("[M+Na]+")
        adduct = Adduct.parse("[2M+H]+")
        adduct = Adduct.parse("[M+H+2i]+")
        adduct = Adduct.parse("[M+H-2H2O]+")
        adduct = Adduct.parse("[M+H-NH3]+")
        adduct = Adduct.parse("[M+2H]2+")
        adduct = Adduct.parse("[M+NH4]+")
        adduct = Adduct.parse("[M+H-CH2O2]+")
        adduct = Adduct.parse("[M+H-CH4O]+")
        adduct = Adduct.parse("[M+H-C4H8]+")
        adduct = Adduct.parse("[M+H-H2O+2i]+")
        adduct = Adduct.parse("[M+H-HF]+")
        adduct = Adduct.parse("[3M+H]+")

        adduct = Adduct.parse("[M-H]-")
        adduct = Adduct.parse("[2M-H]-")
        adduct = Adduct.parse("[M-H-CO2]-")
        adduct = Adduct.parse("[M+CHO2]-")
        adduct = Adduct.parse("[M+Cl]-")
        adduct = Adduct.parse("[M-H+2i]-")
        adduct = Adduct.parse("[M-2H]2-")
        adduct = Adduct.parse("[2M-H+2i]-")
        f = adduct.calc_formula(Formula.parse("C6H12O6"))
        adduct = Adduct.parse("[M-H-C6H10O5]-")
        adduct = Adduct.parse("[2M-H+4i]-")
        adduct = Adduct.parse("[3M-H]-")
        adduct = Adduct.parse("[M-H-HF]-")
        adduct = Adduct.parse("[M-H-SO3]-")
        adduct = Adduct.parse("[M+Br]-")
        

    def test_simple_positive_adduct(self):
        # Test [M+H]+
        adduct = Adduct(ion_type="M", adducts_in=[self.h])
        self.assertEqual(str(adduct), "[M+H]+")
        self.assertEqual(adduct.charge, 1)
        self.assertAlmostEqual(adduct.mass_shift, self.h.exact_mass, places=4)

    def test_simple_negative_adduct(self):
        # Test [M-H]-
        adduct = Adduct(ion_type="M", adducts_out=[self.h])
        self.assertEqual(str(adduct), "[M-H]-")
        self.assertEqual(adduct.charge, -1)
        self.assertAlmostEqual(adduct.mass_shift, -self.h.exact_mass, places=4)

    def test_multiple_adducts(self):
        # Test [M+Na+2H]+
        adduct = Adduct(ion_type="M", adducts_in=[self.na, self.h, self.h])
        self.assertIn("Na", str(adduct))
        self.assertIn("2H", str(adduct))
        self.assertEqual(adduct.charge, 3)
        self.assertTrue(adduct.mass_shift > 0)

    def test_from_str(self):
        # Test parsing from string
        adduct = Adduct.parse("[M+H2O-H]-")
        self.assertEqual(str(adduct), "[M+H2O-H]-")
        self.assertEqual(adduct.charge, -1)
        self.assertIn("O", adduct.element_diff)

    def test_copy(self):
        # Test copy method
        adduct = Adduct(ion_type="M", adducts_in=[self.h])
        copied = adduct.copy()
        self.assertEqual(adduct, copied)
        self.assertIsNot(adduct, copied)

    def test_eq_and_hash(self):
        # Test equality and hashing
        a1 = Adduct.parse("[M+H]+")
        a2 = Adduct.parse("[M+H]+")
        self.assertEqual(a1, a2)
        self.assertEqual(hash(a1), hash(a2))
        self.assertNotEqual(a1, Adduct.parse("[M-H]-"))

    def test_element_diff(self):
        # Test element difference calculation
        adduct = Adduct.parse("[M+Na-H]+")
        diff = adduct.element_diff
        self.assertIn("Na", diff)
        self.assertIn("H", diff)
        self.assertEqual(diff["Na"], 1)
        self.assertEqual(diff["H"], -1)
        self.assertEqual(adduct.charge, 1)

    def test_formula_property(self):
        # [M+H]+ → formula should be H+
        adduct1 = Adduct(ion_type="M", adducts_in=[self.h])
        self.assertEqual(adduct1.formula_shift.value, "H+")
        self.assertEqual(adduct1.formula_shift.charge, 1)

        # [M-H2O] → formula should be -H2O
        adduct2 = Adduct(ion_type="M", adducts_out=[self.h2o])
        self.assertEqual(adduct2.formula_shift.value, "-H2-O")
        self.assertEqual(adduct2.formula_shift.charge, 0)

        # [M+Na+H] → formula should be NaH with +2 charge
        adduct3 = Adduct(ion_type="M", adducts_in=[self.na, self.h])
        self.assertEqual(adduct3.formula_shift.value, "HNa+2")
        self.assertEqual(adduct3.formula_shift.charge, 2)

        # [M+H2O-H2O] → cancels out, formula should be empty
        adduct4 = Adduct(ion_type="M", adducts_in=[self.h2o], adducts_out=[self.h2o])
        self.assertEqual(adduct4.formula_shift.value, "")
        self.assertEqual(adduct4.formula_shift.charge, 0)


if __name__ == "__main__":
    unittest.main()
