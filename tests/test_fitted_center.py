"""Tests for the reported soma center."""

import unittest

import numpy as np

from aind_exaspim_soma_detection.utils.img_util import fitted_center

VOXEL = (100, 200, 300)
SHAPE = (36, 36, 36)


def fit_params(center):
    """
    Returns "fit_rotated_gaussian_3d" parameters with the given center.

    Parameters
    ----------
    center : Tuple[float]
        Center in patch voxel coordinates (z, y, x).

    Returns
    -------
    tuple
        Fit parameters with a round Gaussian at "center".
    """
    return (*center, 0.1, 0, 0, 0.1, 0, 0.1, 100.0, 10.0)


class FittedCenterTest(unittest.TestCase):
    """The fitted center replaces the patch's center voxel."""

    def test_centered_fit(self):
        """A fit at the patch center gives the proposal voxel."""
        center = fitted_center(VOXEL, SHAPE, fit_params((18, 18, 18)))
        np.testing.assert_allclose(center, VOXEL)

    def test_offset_fit(self):
        """The fitted offset is added, in z, y, x order, without rounding."""
        center = fitted_center(VOXEL, SHAPE, fit_params((10.5, 18, 20.25)))
        np.testing.assert_allclose(center, (92.5, 200, 302.25))

    def test_failed_fit(self):
        """A failed fit (all-zero parameters) keeps the proposal voxel."""
        center = fitted_center(VOXEL, SHAPE, np.zeros(11))
        np.testing.assert_allclose(center, VOXEL)

    def test_center_outside_patch(self):
        """A center outside the patch keeps the proposal voxel."""
        for local in ((-1, 18, 18), (18, 18, 40)):
            with self.subTest(local=local):
                center = fitted_center(VOXEL, SHAPE, fit_params(local))
                np.testing.assert_allclose(center, VOXEL)


if __name__ == "__main__":
    unittest.main()
