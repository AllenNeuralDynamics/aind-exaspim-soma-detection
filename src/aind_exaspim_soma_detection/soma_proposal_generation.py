"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

add description 

"""

import numpy as np

from scipy.ndimage import gaussian_laplace, maximum_filter, center_of_mass
from scipy.optimize import curve_fit
from skimage.measure import label


def detect_blobs(
    img_patch, LoG_sigma=2, LoG_threshold=50, bright_threshold=100
):
    LoG_img = gaussian_laplace(img_patch, LoG_sigma)
    LoG_thresholded_img = np.logical_and(
        LoG_img == maximum_filter(LoG_img, LoG_sigma),
        LoG_img > LoG_threshold,
    )
    return np.logical_and(LoG_thresholded_img, img_patch > bright_threshold)


def get_centroids(img, blobs):
    labels, n_labels = label(blobs, return_num=True)
    index = np.arange(1, n_labels + 1)
    centers = center_of_mass(img, labels=labels, index=index)
    return adjust_centers_by_brightness(img, centers)


def adjust_centers_by_brightness(img, centers, k=10):
    """
    Adjust centers in a 3D image to the location of the brightest voxel in a
    local neighborhood.

    Parameters
    ----------
    img : np.ndarray
        A 3D image where voxel intensity values used to identify the brightest
        voxel in a neighborhood.
    centers : list of tuples
        A list of coordinates representing the initial centers to be adjusted.
        Each coordinate is a tuple of integers (x, y, z).
    k : int, optional
        The size of the neighborhood around each center within which to search
        for the brightest voxel. Default is 10.

    Returns
    -------
    list of tuples
        Refined center coordinates where each center has been adjusted to the
        brightest voxel in its neighborhood. Duplicate adjusted centers are
        removed.

    """
    adjusted_centers = set()
    for center in centers:
        voxel = tuple([int(c) for c in center])
        voxel = find_argmax_in_nbhd(img, voxel, k)
        adjusted_centers.add(tuple(voxel))
    return list(adjusted_centers)


def find_argmax_in_nbhd(img, xyz, k):
    # Initializations
    x, y, z = xyz
    half_k = k // 2

    # Neighborhood bounds
    x_min, x_max = max(x - half_k, 0), min(x + half_k + 1, img.shape[0])
    y_min, y_max = max(y - half_k, 0), min(y + half_k + 1, img.shape[1])
    z_min, z_max = max(z - half_k, 0), min(z + half_k + 1, img.shape[2])

    # Find the argmax
    nbhd = img[x_min:x_max, y_min:y_max, z_min:z_max]
    argmax = np.unravel_index(np.argmax(nbhd), nbhd.shape)
    return (argmax[0] + x_min, argmax[1] + y_min, argmax[2] + z_min)


# --- Filter with Brightness ---


# --- Filter with Gaussian Fit ---
def fit_gaussian(img, center, radius):
    # Get patch from img
    x0, y0, z0 = center
    x_min, x_max = max(0, x0 - radius), min(img.shape[0], x0 + radius + 1)
    y_min, y_max = max(0, y0 - radius), min(img.shape[1], y0 + radius + 1)
    z_min, z_max = max(0, z0 - radius), min(img.shape[2], z0 + radius + 1)
    sub_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
    img_vals = sub_img.ravel()

    # Generate coordinates for fitting
    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    z = np.arange(z_min, z_max)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    coords = (x.ravel(), y.ravel(), z.ravel())

    # Fit the Gaussian
    try:
        amplitude = np.max(sub_img)
        offset = np.min(sub_img)
        p0 = (x0, y0, z0, radius, radius, radius, amplitude, offset)
        params, _ = curve_fit(gaussian_3d, coords, img_vals, p0=p0)
        return img_vals, coords, params
    except RuntimeError as e:
        return None


def fitness_quality(img, coords, params):
    fitted_gaussian = gaussian_3d(coords, *params).reshape(img.shape)
    fitted = fitted_gaussian.flatten()
    actual = img.flatten()
    return np.corrcoef(actual, fitted)[0, 1]


def gaussian_3d(
    xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, amplitude, offset
):
    x, y, z = xyz
    value = (amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) +
          ((y - y0) ** 2) / (2 * sigma_y ** 2) +
          ((z - z0) ** 2) / (2 * sigma_z ** 2))
    ) + offset).ravel()
    return value

