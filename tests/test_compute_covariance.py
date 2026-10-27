"""Tests for the physical covariance of fitted somas."""

import unittest

import numpy as np

from aind_exaspim_soma_detection.soma_proposal_classification import (
    compute_covariance,
    compute_radii,
    covariance_entries,
)

#: Voxel size at multiscale level 2, ordered z, y, x like an image patch.
VOXEL_ZYX = np.array([4.0, 2.992, 2.992])


def fit_params(cov_xyz):
    """
    Returns "fit_rotated_gaussian_3d" parameters for a Gaussian with the
    given physical covariance.

    Parameters
    ----------
    cov_xyz : numpy.ndarray
        3x3 covariance in square microns, axes ordered x, y, z.

    Returns
    -------
    tuple
        Center, precision matrix entries, amplitude and offset, in the
        patch's voxel coordinates (axes z, y, x).
    """
    S_inv = np.diag(1 / VOXEL_ZYX)
    P = np.linalg.inv(S_inv @ cov_xyz[::-1, ::-1] @ S_inv)
    return (20, 20, 20, P[0, 0], P[0, 1], P[0, 2], P[1, 1], P[1, 2],
            P[2, 2], 1, 0)


def rotated_cov(angle_deg, sigma_xyz):
    """
    Returns a covariance rotated in the x-y plane.

    Parameters
    ----------
    angle_deg : float
        Angle of the first principal axis from x, towards y.
    sigma_xyz : Tuple[float]
        Standard deviations along the principal axes before rotation.

    Returns
    -------
    numpy.ndarray
        3x3 covariance in square microns, axes ordered x, y, z.
    """
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0],
                  [0, 0, 1]])
    return R @ np.diag(np.square(sigma_xyz)) @ R.T


class ComputeCovarianceTest(unittest.TestCase):
    """The covariance is recovered in microns, axes x, y, z."""

    def test_rotated_soma(self):
        """A soma elongated 30 degrees from x keeps its orientation."""
        cov = rotated_cov(30, (14, 8, 8))
        result = compute_covariance(fit_params(cov), multiscale=2)
        np.testing.assert_allclose(result, cov, atol=1e-9)
        values, vectors = np.linalg.eigh(result)
        long_axis = vectors[:, np.argmax(values)]
        expected = [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0]
        self.assertAlmostEqual(abs(long_axis @ expected), 1.0)

    def test_radii_match_covariance(self):
        """The radii are 2 * sqrt of the covariance's eigenvalues."""
        params = fit_params(rotated_cov(30, (14, 8, 8)))
        np.testing.assert_allclose(compute_radii(params, 2), [16, 16, 28])

    def test_singular_fit(self):
        """A singular precision matrix gives no covariance and no radii."""
        params = (20, 20, 20, 0, 0, 0, 0, 0, 0, 1, 0)
        self.assertIsNone(compute_covariance(params, 2))
        self.assertEqual(compute_radii(params, 2).size, 0)

    def test_entries(self):
        """The six unique entries are written as xx, xy, xz, yy, yz, zz."""
        cov = np.array([[4.0, 1.0, 2.0], [1.0, 9.0, 3.0], [2.0, 3.0, 16.0]])
        self.assertEqual(covariance_entries(cov),
                         (4.0, 1.0, 2.0, 9.0, 3.0, 16.0))
        self.assertIsNone(covariance_entries(None))


if __name__ == "__main__":
    unittest.main()
