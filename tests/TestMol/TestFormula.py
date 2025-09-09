import unittest
from collections import OrderedDict
from rdkit import Chem
from MassMolKit.Mol.Formula import Formula


class TestFormula(unittest.TestCase):

    def setUp(self):
        # Prepare test formulas with expected exact masses
        self.formulas = {
            "C6H12O6": {
                'formula': Formula(OrderedDict([("C", 6), ("H", 12), ("O", 6)]),
                                   charge=0, raw_formula="C6H12O6"),
                'exact_mass': 180.06339,
                'value': "C6H12O6",
                'plain': "C6H12O6",
                'test_str': "C6H12O6",
            },
            "H2O": {
                'formula': Formula(OrderedDict([("H", 2), ("O", 1)]),
                                   charge=0, raw_formula="H2O"),
                'exact_mass': 18.01056,
                'value': "H2O",
                'plain': "H2O",
                'test_str': "HOH",
            },
            "Na+": {
                'formula': Formula(OrderedDict([("Na", 1)]),
                                   charge=1, raw_formula="Na+"),
                'exact_mass': 22.98977,
                'value': "Na+",
                'plain': "Na",
                'test_str': "Na+",
            },
            "-C-H4-O": {
                'formula': Formula(OrderedDict([("C", -1), ("H", -4), ("O", -1)]),
                                   charge=0, raw_formula="-(CH3OH)"),
                'exact_mass': -32.02621,
                'value': "-C-H4-O",
                'plain': "-C-H4-O",
                'test_str': "-H5-C-OH",
            },
            "C6H12O6-Na-": {    # Glucose minus sodium
                'formula': Formula(OrderedDict([("C", 6), ("H", 12), ("O", 6), ("Na", -1)]),
                                   charge=-1, raw_formula="C6H12O6-Na-"),
                'exact_mass': 157.07362,
                'value': "C6H12-NaO6-",
                'plain': "C6H12-NaO6",
                'test_str': "C6H12-NaO6-",
            },
        }

    def test_exact_mass(self):
        # Check whether the exact mass is correctly calculated
        for name, entry in self.formulas.items():
            calc_mass = round(entry['formula'].exact_mass, 5)
            expected = round(entry['exact_mass'], 5)
            self.assertAlmostEqual(calc_mass, expected, places=4,
                                   msg=f"Mismatch in exact_mass for {name}")

    def test_to_string_and_value_plain(self):
        # Test string representations with and without charges
        for name, entry in self.formulas.items():
            f = entry['formula']
            self.assertEqual(f.to_string(no_charge=False), entry['value'],
                             msg=f"Mismatch in to_string(no_charge=False) for {name}")
            self.assertEqual(f.to_string(no_charge=True), entry['plain'],
                                msg=f"Mismatch in to_string(no_charge=True) for {name}")
            self.assertEqual(f.value, entry['value'],
                                msg=f"Mismatch in value property for {name}")
            self.assertEqual(f.plain, entry['plain'],
                                msg=f"Mismatch in plain property for {name}")
            

    def test_reorder_elements(self):
        # Even if the input order is different, Hill system order should be applied
        f = Formula(OrderedDict([("O", 6), ("C", 6), ("H", 12)]),
                    charge=0, raw_formula="unordered")
        
        self.assertEqual(list(f._elements.keys())[0], "C")  # "C" should appear first in Hill order
        self.assertEqual(list(f._elements.keys())[1], "H")
        self.assertEqual(list(f._elements.keys())[2], "O")

    def test_copy(self):
        # Copy should create an equal but independent instance
        f = self.formulas["C6H12O6"]["formula"]
        f_copy = f.copy()
        self.assertEqual(f, f_copy)
        self.assertIsNot(f, f_copy)
        self.assertEqual(f.elements, f_copy.elements)
        self.assertEqual(f.charge, f_copy.charge)

    def test_from_str(self):
        for name, entry in self.formulas.items():
            f_parsed = Formula.from_str(entry['test_str'])
            self.assertEqual(f_parsed, entry['formula'],
                             msg=f"Mismatch in from_str for {name}")
            
    def test_addition(self):
        # Glucose + Water -> C6H14O7
        glucose = self.formulas["C6H12O6"]["formula"]
        water = self.formulas["H2O"]["formula"]
        result = glucose + water
        self.assertEqual(result.value, "C6H14O7",
                         msg="Mismatch in addition: Glucose + Water")

        # Sodium ion + Water -> H2NaO+
        sodium = self.formulas["Na+"]["formula"]
        result2 = sodium + water
        self.assertEqual(result2.value, "H2NaO+",
                         msg="Mismatch in addition: Na+ + H2O")
        
        # Negative group (-CH3OH) + Water -> -C-H2O2
        negative = self.formulas["-C-H4-O"]["formula"]
        result3 = negative + water
        self.assertEqual(result3.value, "-C-H2",
                         msg="Mismatch in addition: -(CH3OH) + H2O")
        


    def test_subtraction(self):
        # Glucose - Water -> C6H10O5
        glucose = self.formulas["C6H12O6"]["formula"]
        water = self.formulas["H2O"]["formula"]
        result = glucose - water
        self.assertEqual(result.value, "C6H10O5",
                         msg="Mismatch in subtraction: Glucose - Water")

        # Glucose-Na- - Na+ -> C6H12O6-NaNa-
        glucose_na = self.formulas["C6H12O6-Na-"]["formula"]
        sodium = self.formulas["Na+"]["formula"]
        result2 = glucose_na - sodium
        self.assertEqual(result2.value, "C6H12-Na2O6-2",
                         msg="Mismatch in subtraction: Glucose-Na- - Na+")
        
        # Water - Negative group (-CH3OH) -> C-H2O2
        negative = self.formulas["-C-H4-O"]["formula"]
        result3 = water - negative
        self.assertEqual(result3.value, "CH6O2",
                         msg="Mismatch in subtraction: H2O - (-(CH3OH))")


if __name__ == "__main__":
    unittest.main()
