"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import rotate

import numpy as np
import random


class ImageTransforms:
    """
    Class that applies a sequence of transforms to a 3D image and segmentation
    patch.
    """

    def __init__(self):
        """
        Initializes an ImageTransforms instance that applies augmentation to
        an image and segmentation patch.
        """
        # Instance attributes
        self.transforms = [
            RandomFlip3D(),
            RandomRotation3D(),
            RandomNoise3D(),
            RandomContrast3D()
        ]

    def __call__(self, img):
        """
        Applies geometric transforms to an image.

        Parameters
        ----------
        img : numpy.ndarray
            Image with shape (H, W, D) to be transformed.
        """
        for transform in self.transforms:
            img = transform(img)
        return img


# --- Geometric Transforms ---
class RandomFlip3D:
    """
    Randomly flips a 3D image along one or more axes.
    """

    def __init__(self, axes=(0, 1, 2)):
        """
        Initializes a RandomFlip3D transformer.

        Parameters
        ----------
        axes : Tuple[float], optional
            Axes along which to flip the image. Default is (0, 1, 2).
        """
        self.axes = axes

    def __call__(self, img):
        """
        Applies random flipping to the input image and segmentation patch.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be flipped.
        """
        for axis in self.axes:
            if random.random() > 0.5:
                img = np.flip(img, axis=axis)
        return img


class RandomRotation3D:
    """
    Applies random rotation along a randomly chosen axis.
    """

    def __init__(self, angles=(-90, 90), axes=((0, 1), (0, 2), (1, 2))):
        """
        Initializes a RandomRotation3D transformer.

        Parameters
        ----------
        angles : Tuple[int], optional
            Maximum angle of rotation. Default is (-45, 45).
        axis : Tuple[Tuple[int]], optional
            Axes to apply rotation. Default is ((0, 1), (0, 2), (1, 2))
        """
        self.angles = angles
        self.axes = axes

    def __call__(self, img):
        """
        Rotates the input image and segmentation patch.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be rotated.
        """
        for axes in self.axes:
            if random.random() < 0.5:
                angle = random.uniform(*self.angles)
                img = rotate(
                    img,
                    angle,
                    axes=axes,
                    mode="grid-mirror",
                    reshape=False,
                )
        return img


# --- Intensity Transforms ---
class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.
    """

    def __init__(self, p_low=(0, 90), p_high=(97.5, 100)):
        """
        Initializes a RandomContrast3D transformer.

        Parameters
        ----------
        ...
        """
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, img):
        """
        Applies contrast to the input 3D image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be contrasted.
        """
        lo = np.percentile(img, np.random.uniform(*self.p_low))
        hi = np.percentile(img, np.random.uniform(*self.p_high))
        img = (img - lo) / (hi - lo + 1e-5)
        img = np.clip(img, 0, 1)
        return img


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.
    """

    def __init__(self, max_std=0.2):
        """
        Initializes a RandomNoise3D transformer.

        Parameters
        ----------
        max_std : float, optional
            Maximum standard deviation of the Gaussian noise distribution.
            Default is 0.3.
        """
        self.max_std = max_std

    def __call__(self, img):
        """
        Adds Gaussian noise to the input 3D image.

        Parameters
        ----------
        patches : numpy.ndarray
            Image to add noise to.
        """
        std = self.max_std * random.random()
        img += np.random.uniform(-std, std, img.shape)
        return img
