"""Tests for the soma radius feasibility check."""

import unittest

import numpy as np

from aind_exaspim_soma_detection.soma_proposal_classification import (
    has_feasible_radii,
)


class HasFeasibleRadiiTest(unittest.TestCase):
    """Every radius must be strictly between the bounds."""

    def test_plausible_radii(self):
        """Three radii within the bounds are feasible."""
        self.assertTrue(has_feasible_radii(np.array([10.0, 20.0, 40.0])))

    def test_any_radius_out_of_bounds(self):
        """One radius outside the bounds makes the fit infeasible."""
        for radii in ([5.0, 20.0, 40.0], [10.0, 20.0, 400.0],
                      [6.0, 20.0, 40.0], [10.0, 20.0, 140.0]):
            with self.subTest(radii=radii):
                self.assertFalse(has_feasible_radii(np.array(radii)))

    def test_failed_fit(self):
        """No radii (a failed conversion) or NaN radii are infeasible."""
        self.assertFalse(has_feasible_radii(np.zeros([0, 0, 0])))
        self.assertFalse(has_feasible_radii(np.array([np.nan, 20.0, 40.0])))

    def test_custom_bounds(self):
        """The bounds can be changed."""
        radii = np.array([4.0, 20.0, 40.0])
        self.assertTrue(has_feasible_radii(radii, min_radius=2))


if __name__ == "__main__":
    unittest.main()
