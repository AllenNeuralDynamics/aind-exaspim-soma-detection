"""Tests for the physical radii of fitted somas."""

import unittest

import numpy as np

from aind_exaspim_soma_detection.soma_proposal_classification import (
    compute_radii,
)

#: Voxel size at multiscale level 2, ordered z, y, x like an image patch.
VOXEL_ZYX = np.array([4.0, 2.992, 2.992])


def fit_params(sigma_zyx):
    """
    Returns "fit_rotated_gaussian_3d" parameters for an axis-aligned
    Gaussian with the given physical standard deviations.

    Parameters
    ----------
    sigma_zyx : Tuple[float]
        Standard deviation in microns along z, y, x.

    Returns
    -------
    tuple
        Center, precision matrix entries, amplitude and offset, in the
        patch's voxel coordinates (axes z, y, x).
    """
    a11, a22, a33 = (VOXEL_ZYX / np.asarray(sigma_zyx, dtype=float)) ** 2
    return (20, 20, 20, a11, 0, 0, a22, 0, a33, 1, 0)


class ComputeRadiiTest(unittest.TestCase):
    """Radii must be in microns along the correct axes."""

    def test_round_soma(self):
        """A round soma has three equal radii."""
        radii = compute_radii(fit_params((10, 10, 10)), multiscale=2)
        np.testing.assert_allclose(radii, [20, 20, 20])

    def test_elongated_soma(self):
        """Elongation along z and along x gives the same radii."""
        for sigma_zyx in ((14, 8, 8), (8, 8, 14)):
            with self.subTest(sigma_zyx=sigma_zyx):
                radii = compute_radii(fit_params(sigma_zyx), multiscale=2)
                np.testing.assert_allclose(radii, [16, 16, 28])


if __name__ == "__main__":
    unittest.main()
