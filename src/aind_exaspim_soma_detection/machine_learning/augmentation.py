"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import rotate, zoom

import numpy as np
import random


class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.
    """

    def __init__(self, factor_range=(0.8, 1.2)):
        """
        Initializes a RandomContrast3D transformer.

        Parameters
        ----------
        factor_range : Tuple[float], optional
            Tuple of integers representing the range of contrast factors.
            Default is (0.8, 1.1).
        """
        self.factor_range = factor_range

    def __call__(self, img):
        """
        Applies contrast to the input 3D image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to which contrast will be added.

        Returns
        -------
        numpy.ndarray
            Contrasted 3D image.
        """
        factor = random.uniform(*self.factor_range)
        return np.clip(img * factor, img.min(), img.max())


class RandomFlip3D:
    """
    Randomly flip a 3D image along one or more axes.
    """

    def __init__(self, axes=(0, 1, 2)):
        """
        Initializes a RandomFlip3D transformer.

        Parameters
        ----------
        axes : Tuple[float], optional
            Tuple of integers representing the axes along which to flip the
            image. Default is (0, 1, 2).
        """
        self.axes = axes

    def __call__(self, img):
        """
        Applies random flipping to the input 3D image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be flipped.

        Returns
        -------
        numpy.ndarray
            Flipped 3D image.
        """
        for axis in self.axes:
            if random.random() > 0.5:
                img = np.flip(img, axis=axis)
        return img


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.
    """

    def __init__(self, mean=0.0, std=0.025):
        """
        Initializes a RandomNoise3D transformer.

        Parameters
        ----------
        mean : float, optional
            Mean of the Gaussian noise distribution. The default is 0.0.
        std : float, optional
            Standard deviation of the Gaussian noise distribution. Default is
            0.025.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Adds Gaussian noise to the input 3D image.

        Parameters
        ----------
        img : np.ndarray
            Image to which noise will be added.

        Returns
        -------
        numpy.ndarray
            Noisy 3D image.
        """
        noise = np.random.normal(self.mean, self.std, img.shape)
        return img + noise


class RandomRotation3D:
    """
    Applies random rotation to a 3D image along a randomly chosen axis.
    """

    def __init__(self, angles=(-45, 45), axes=((0, 1), (0, 2), (1, 2))):
        """
        Initializes a RandomRotation3D transformer.

        Parameters
        ----------
        angles : Tuple[int], optional
            Maximum angle of rotation. Default is (-45, 45).
        axis : Tuple[Tuple[int]], optional
            Axes to apply rotation.
        """
        self.angles = angles
        self.axes = axes

    def __call__(self, img, mode="grid-mirror"):
        """
        Rotates the input 3D image.

        Parameters
        ----------
        img : np.ndarray
            Image to be rotated.
        mode : str, optional
            Method of extrapolating image after rotation. Default is
            "grid-mirror".

        Returns
        -------
        numpy.ndarray
            Rotated 3D image.
        """
        for axis in self.axes:
            angle = random.uniform(*self.angles)
            img = rotate(
                img, angle, axes=axis, mode=mode, reshape=False, order=1
            )
        return img


class RandomScale3D:
    """
    Applies random scaling to a 3D image along each axis.
    """

    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Initializes a RandomScale3D transformer.

        Parameters
        ----------
        scale_range : Tuple[float], optional
            Range of scaling factors. Default is (0.9, 1.1).
        """
        self.scale_range = scale_range

    def __call__(self, img):
        """
        Applies random scaling to the input 3D image.

        Parameters
        ----------
        img : np.ndarray
            Image to be scaled.

        Returns
        -------
        numpy.ndarray
            Scaled 3D image.
        """
        # Sample new image shape
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_shape = (
            int(img.shape[0] * alpha),
            int(img.shape[1] * alpha),
            int(img.shape[2] * alpha),
        )

        # Compute the zoom factors
        shape = img.shape
        zoom_factors = [
            new_dim / old_dim for old_dim, new_dim in zip(shape, new_shape)
        ]
        return zoom(img, zoom_factors, order=3)
