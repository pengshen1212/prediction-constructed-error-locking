import unittest
import numpy as np
from reanalyze_mechanism import summarize

class ReanalysisTests(unittest.TestCase):
    def test_realized_weights_not_nominal_weights(self):
        cell = dict(phat=[0.2, 0.8], mA=[0.6, 1.7], mB=[0, 0],
                    mA_cor=[1, 2], mB_cor=[0, 0],
                    mA_cln=[0.5, 0.5], mB_cln=[0, 0],
                    aA=[0.2, 0.8], aB=[0.1, 0.6])
        r = summarize(cell, n_boot=100)
        self.assertAlmostEqual(r["corrupted_contribution"], 0.9)
        self.assertAlmostEqual(r["clean_contribution"], 0.25)
        self.assertAlmostEqual(r["margin_difference"], 1.15)
        self.assertLess(r["max_decomposition_residual"], 1e-12)

    def test_pairing_is_preserved(self):
        a = np.array([10., 20., 30.])
        cell = dict(phat=[0.5]*3, mA=a, mB=a-1,
                    mA_cor=a, mB_cor=a-1, mA_cln=a, mB_cln=a-1,
                    aA=[0.7]*3, aB=[0.6]*3)
        r = summarize(cell, n_boot=100)
        np.testing.assert_allclose(r["margin_ci"], [1, 1])

    def test_inconsistent_record_is_rejected(self):
        cell = dict(phat=[0.5]*2, mA=[1, 1], mB=[0, 0],
                    mA_cor=[0, 0], mB_cor=[0, 0],
                    mA_cln=[0, 0], mB_cln=[0, 0],
                    aA=[1, 1], aB=[0, 0])
        with self.assertRaises(AssertionError):
            summarize(cell, n_boot=100)

if __name__ == "__main__":
    unittest.main()

