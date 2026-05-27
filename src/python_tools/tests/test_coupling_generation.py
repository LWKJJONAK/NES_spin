from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Ising2d_power_law import power_law_coupling_from_positions
from generate_trapped_ion_coupling import coupling_for_mode


class CouplingGenerationTests(unittest.TestCase):
    def test_power_law_coupling_from_positions(self) -> None:
        r = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 3.0],
            ]
        )

        coupling = power_law_coupling_from_positions(r, j0=1.0, alpha=1.0)

        expected = np.array(
            [
                [0.0, 4.0 / 3.0, 4.0 / 9.0],
                [4.0 / 3.0, 0.0, 2.0 / 3.0],
                [4.0 / 9.0, 2.0 / 3.0, 0.0],
            ]
        )
        np.testing.assert_allclose(coupling, expected)

    def test_single_mode_coupling_is_symmetric_and_zero_diagonal(self) -> None:
        omega_k = np.array([1.0, 2.0, 3.0]) * 2 * np.pi
        b_jk = np.eye(3)

        coupling = coupling_for_mode(omega_k, b_jk, mode=1, detuning_mhz=0.001, rabi_mhz=0.010)

        np.testing.assert_allclose(coupling, coupling.T)
        np.testing.assert_allclose(np.diag(coupling), 0.0)
        self.assertEqual(coupling.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
