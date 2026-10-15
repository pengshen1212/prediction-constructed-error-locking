import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


os.environ.setdefault("PAPER2_AUDIT_OUT", tempfile.gettempdir())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_prognn import _to_numpy


class ToNumpyTests(unittest.TestCase):
    def test_handles_tensor_scipy_sparse_and_array(self):
        cases = [
            torch.tensor([[1.0, 2.0]]),
            sp.csr_matrix([[1.0, 2.0]]),
            np.array([[1.0, 2.0]]),
        ]
        for value in cases:
            with self.subTest(type=type(value).__name__):
                np.testing.assert_array_equal(_to_numpy(value), [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
