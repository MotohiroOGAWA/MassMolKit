import unittest
import math

from mmkit.mass.Tolerance import *

class TestDaTolerance(unittest.TestCase):
    def test_unit(self):
        tol = DaTolerance(0.01)
        self.assertEqual(tol.unit, "Da")

    def test_error(self):
        tol = DaTolerance(0.01)
        self.assertAlmostEqual(tol.error(100.005, 100.000), 0.005, places=12)

    def test_within_true(self):
        tol = DaTolerance(0.01)
        self.assertTrue(tol.within(100.009, 100.000))

    def test_within_false(self):
        tol = DaTolerance(0.01)
        self.assertFalse(tol.within(100.011, 100.000))

    def test_ops(self):
        tol = DaTolerance(0.01)
        self.assertIsInstance(tol + 0.01, DaTolerance)
        self.assertAlmostEqual((tol + 0.01).tolerance, 0.02, places=12)
        self.assertAlmostEqual((tol - 0.005).tolerance, 0.005, places=12)
        self.assertAlmostEqual((tol * 2).tolerance, 0.02, places=12)
        self.assertAlmostEqual((tol / 2).tolerance, 0.005, places=12)


class TestPpmTolerance(unittest.TestCase):
    def test_unit(self):
        tol = PpmTolerance(10.0)
        self.assertEqual(tol.unit, "ppm")

    def test_error(self):
        tol = PpmTolerance(10.0)
        # observed - theoretical = 0.001 at theoretical 100.0 -> 10 ppm
        self.assertAlmostEqual(tol.error(100.001, 100.0), 10.0, places=9)

    def test_within_true(self):
        tol = PpmTolerance(10.1)
        self.assertTrue(tol.within(100.001, 100.0))

    def test_within_false(self):
        tol = PpmTolerance(10.0)
        self.assertFalse(tol.within(100.0011, 100.0))  # 11 ppm

    def test_ops(self):
        tol = PpmTolerance(10.0)
        self.assertIsInstance(tol + 5.0, PpmTolerance)
        self.assertAlmostEqual((tol + 5.0).tolerance, 15.0, places=12)
        self.assertAlmostEqual((tol - 2.0).tolerance, 8.0, places=12)
        self.assertAlmostEqual((tol * 2).tolerance, 20.0, places=12)
        self.assertAlmostEqual((tol / 4).tolerance, 2.5, places=12)


class TestSwitchingWithinTolerance(unittest.TestCase):
    def test_mode_da_sets_tolerance_to_da_within(self):
        tol = SwitchingWithinTolerance(mode="Da", da_within=0.01, ppm_within=10.0, switch_mass=500.0)
        self.assertEqual(tol.unit, "Da")
        self.assertAlmostEqual(tol.tolerance, 0.01, places=12)

    def test_mode_ppm_sets_tolerance_to_ppm_within(self):
        tol = SwitchingWithinTolerance(mode="ppm", da_within=0.01, ppm_within=10.0, switch_mass=500.0)
        self.assertEqual(tol.unit, "ppm")
        self.assertAlmostEqual(tol.tolerance, 10.0, places=12)

    def test_error_uses_mode_da(self):
        tol = SwitchingWithinTolerance(mode="Da", da_within=0.01, ppm_within=10.0, switch_mass=500.0)
        self.assertAlmostEqual(tol.error(100.005, 100.000), 0.005, places=12)

    def test_error_uses_mode_ppm(self):
        tol = SwitchingWithinTolerance(mode="ppm", da_within=0.01, ppm_within=10.0, switch_mass=500.0)
        # observed - theoretical = 0.001 at theoretical 100 -> 10 ppm
        self.assertAlmostEqual(tol.error(100.001, 100.0), 10.0, places=9)

    def test_within_switches_to_da_below_threshold(self):
        tol = SwitchingWithinTolerance(mode="ppm", da_within=0.01, ppm_within=10.0, switch_mass=500.0)
        theoretical = 100.0  # below 500 -> Da rule
        self.assertTrue(tol.within(theoretical + 0.009, theoretical))
        self.assertFalse(tol.within(theoretical + 0.011, theoretical))

    def test_within_switches_to_ppm_at_or_above_threshold(self):
        tol = SwitchingWithinTolerance(mode="Da", da_within=0.01, ppm_within=10.0, switch_mass=500.0)
        theoretical = 500.0  # >= 500 -> ppm rule (note: code uses < for Da branch)
        # 10 ppm at 500.0 => 0.005 Da
        self.assertTrue(tol.within(theoretical + 0.005, theoretical))
        self.assertFalse(tol.within(theoretical + 0.006, theoretical))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            SwitchingWithinTolerance(mode="xyz", da_within=0.01, ppm_within=10.0, switch_mass=500.0)

    def test_parse_switch_da_mode(self):
        tol = parse_mass_tolerance("switch:0.01da,10ppm@500", "da")
        self.assertIsInstance(tol, SwitchingWithinTolerance)
        self.assertEqual(tol.unit, "Da")
        self.assertAlmostEqual(tol._da_within.tolerance, 0.01, places=12)
        self.assertAlmostEqual(tol._ppm_within.tolerance, 10.0, places=12)
        self.assertAlmostEqual(tol.switch_mass, 500.0, places=12)
        # mode=Da => tolerance should equal da_within
        self.assertAlmostEqual(tol.tolerance, 0.01, places=12)

    def test_parse_switch_ppm_mode(self):
        tol = parse_mass_tolerance("switch:0.01da,10ppm@500", "ppm")
        self.assertIsInstance(tol, SwitchingWithinTolerance)
        self.assertEqual(tol.unit, "ppm")
        # mode=ppm => tolerance should equal ppm_within
        self.assertAlmostEqual(tol.tolerance, 10.0, places=12)

    def test_parse_invalid_unit_raises(self):
        with self.assertRaises(ValueError):
            parse_mass_tolerance("switch:0.01da,10ppm@500", "mDa")

    def test_parse_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            parse_mass_tolerance("0.01da,10ppm@500", "ppm")  # missing "switch:"

    def test_parse_negative_values_raise(self):
        with self.assertRaises(ValueError):
            parse_mass_tolerance("switch:-0.01da,10ppm@500", "ppm")
        with self.assertRaises(ValueError):
            parse_mass_tolerance("switch:0.01da,-10ppm@500", "ppm")
        with self.assertRaises(ValueError):
            parse_mass_tolerance("switch:0.01da,10ppm@-500", "ppm")


if __name__ == "__main__":
    unittest.main()
