"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

"""

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_laplace, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from skimage.measure import label

from aind_exaspim_soma_detection.utils import img_util

BRIGHT_THRESHOLD = 80
LOG_SIGMA = 5
LOG_THRESHOLD = 10


# --- Core Routines ---
def generate_proposals(
    img,
    offset,
    margin,
    window_size,
    downsample_factor,
    bright_threshold=BRIGHT_THRESHOLD,
    LoG_sigma=LOG_SIGMA,
    LoG_threshold=LOG_THRESHOLD,
):
    # Read patch
    img_patch = get_patch(img, offset, window_size, from_center=False)
    if np.max(img_patch) < bright_threshold:
        return list(), list()

    # Detect candidates
    blobs = detect_blobs(
        img_patch,
        bright_threshold=bright_threshold,
        LoG_sigma=LoG_sigma,
        LoG_threshold=LoG_threshold,
    )
    centers = get_centers(img_patch, blobs)
    centers = filter_centers(img_patch, centers, margin)

    # Convert coordinates
    physical_centers = list()
    for voxel in centers:
        physical_centers.append(
            img_util.local_to_physical(voxel[::-1], offset, downsample_factor)
        )
    return physical_centers


def detect_blobs(
    img_patch,
    bright_threshold=BRIGHT_THRESHOLD,
    LoG_sigma=LOG_SIGMA,
    LoG_threshold=LOG_THRESHOLD,
):
    LoG_img = gaussian_laplace(img_patch, LoG_sigma)
    LoG_thresholded_img = np.logical_and(
        LoG_img == maximum_filter(LoG_img, LoG_sigma),
        LoG_img > LoG_threshold,
    )
    return np.logical_and(LoG_thresholded_img, img_patch > bright_threshold)


def get_centers(img, blobs):
    labels, n_labels = label(blobs, return_num=True)
    index = np.arange(1, n_labels + 1)
    centers = center_of_mass(img, labels=labels, index=index)
    return adjust_centers_by_brightness(img, centers)


# --- Postprocess Centers ---
def adjust_centers_by_brightness(img, centers, k=6):
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


def find_argmax_in_nbhd(img, voxel, n):
    """
    Finds the coordinate of the maximum value within an n x n x n neighborhood
    of a 3D image centered at the given coordinates.

    Parameters:
    -----------
    img : numpy.ndarray
        A 3D array representing an image.
    voxel : Tuple[int]
        Center coordinate of the neighborhood.
    n : int
        Size of the neighborhood in each dimension.

    Returns:
    --------
    Tuple[int]
        The voxel coordinate with the maximum value within the neighborhood.
    """
    # Initializations
    i, j, k = voxel
    half_n = n // 2

    # Neighborhood bounds
    i_min, i_max = max(i - half_n, 0), min(i + half_n + 1, img.shape[0])
    j_min, j_max = max(j - half_n, 0), min(j + half_n + 1, img.shape[1])
    k_min, k_max = max(k - half_n, 0), min(k + half_n + 1, img.shape[2])

    # Find the argmax
    nbhd = img[i_min:i_max, j_min:j_max, k_min:k_max]
    argmax = np.unravel_index(np.argmax(nbhd), nbhd.shape)
    return (argmax[0] + i_min, argmax[1] + j_min, argmax[2] + k_min)


# --- Filtering ---
def filter_centers(img_patch, centers, margin):
    # Initializations
    brightness = [img_patch[c] for c in centers]
    if len(centers) > 0:
        kdtree = KDTree(centers)
    else:
        return list()

    # Main
    filtered_centers = list()
    visited = set()
    for idx in np.argsort(brightness)[::-1]:
        # Determine whether to visit center
        inbounds_bool = is_inbounds(img_patch, centers[idx], margin)
        not_visited_bool = centers[idx] not in visited
        if inbounds_bool and not_visited_bool:
            center = tuple([int(v) for v in centers[idx]])
            if check_gaussian_fit(img_patch, center) is not None:
                discard_nearby_centers(kdtree, visited, center)
                filtered_centers.append(center)
    return filtered_centers


def check_gaussian_fit(img, voxel, radius=6):
    """
    Fits a Gaussian to the neighborhood of a voxel and evaluates the quality
    of the fit.

    Parameters:
    -----------
    img : numpy.ndarray
        A 3D array representing the image to be analyzed.
    voxel : Tuple[int]
        Center coordinate of the neighborhood where the Gaussian is to be fit.
    radius : int, optional
        Radius of the cubic neighborhood around the center. The default is 6.

    Returns:
    --------
    numpy.ndarray or None
        If the Gaussian fit is of acceptable quality (fitness score ≥ 0.7),
        returns the parameters of the Gaussian. Otherwise, returns None.

    """
    img_vals, coords, params = fit_gaussian(img, voxel, radius)
    if params is not None:
        if fitness_quality(img_vals, coords, params) < 0.7:
            params = None
    return params


def fit_gaussian(img, voxel, radius):
    """
    Fits a 3D Gaussian to the neighborhood of a voxel in a 3D image.

    Parameters:
    -----------
    img : numpy.ndarray
        A 3D array representing an image in which the Gaussian fit is
        performed.
    voxel : Tuple[int]
        Voxel coordiante specifying the center of the neighborhood.
    radius : int
        Radius of the cubic neighborhood around the center voxel. The
        neighborhood size is (2 * radius + 1)³.

    Returns:
    --------
    tuple
        A tuple containing:
            - Flattened array of image values in the neighborhood.
            - Flattened arrays of the voxel coordinates in the neighborhood.
            - Parameters of the fitted Gaussian.

    """
    # Get patch from img
    x0, y0, z0 = voxel
    x_min, x_max = max(0, x0 - radius), min(img.shape[0], x0 + radius + 1)
    y_min, y_max = max(0, y0 - radius), min(img.shape[1], y0 + radius + 1)
    z_min, z_max = max(0, z0 - radius), min(img.shape[2], z0 + radius + 1)
    sub_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
    img_vals = sub_img.ravel()

    # Generate coordinates for fitting
    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    z = np.arange(z_min, z_max)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    coords = (x.ravel(), y.ravel(), z.ravel())

    # Fit the Gaussian
    try:
        amplitude = np.max(sub_img)
        offset = np.min(sub_img)
        p0 = (x0, y0, z0, radius, radius, radius, amplitude, offset)
        params, _ = curve_fit(gaussian_3d, coords, img_vals, p0=p0)
        return img_vals, coords, params
    except RuntimeError:
        return None, None, None


def fitness_quality(img, coords, params):
    """
    Evaluates the quality of a Gaussian fit by computing the correlation
    coefficient between the image values fitted Gaussian values.

    Parameters:
    -----------
    img : numpy.ndarray
        A 3D array representing an image.
    coords : Tuple[numpy.ndarray]
        Flattened arrays of image coordinates.
    params : numpy.ndarray
        Parameters of the fitted Gaussian function.

    Returns:
    --------
    float
        Pearson correlation coefficient between the image values and fitted
        Gaussian values. A value closer to 1 indicates a better fit.

    """
    fitted_gaussian = gaussian_3d(coords, *params).reshape(img.shape)
    fitted = fitted_gaussian.flatten()
    actual = img.flatten()
    return np.corrcoef(actual, fitted)[0, 1]


def global_filtering(xyz_list):
    """
    Filters a list of 3D points by merging nearby points based on a specified
    distance threshold.

    Parameters:
    -----------
    xyz_list : list of tuple
        List of xyz coordinates representing points in 3D space.

    Returns:
    --------
    List[tuple]
        Filtered list of the given 3D points.

    """
    kdtree = KDTree(xyz_list)
    xyz_set = set(xyz_list)
    for i, j in kdtree.query_pairs(5):
        # Extract xyz_list
        xyz_i = kdtree.data[i]
        xyz_j = kdtree.data[j]
        xyz = tuple([(xyz_i[n] + xyz_j[n]) / 2 for n in range(3)])

        # Update xyz_list
        xyz_set.discard(tuple([int(x) for x in xyz_i]))
        xyz_set.discard(tuple([int(x) for x in xyz_j]))
        xyz_set.add(tuple([int(x) for x in xyz]))
    return list(xyz_set)


# --- utils ---
def discard_nearby_centers(kdtree, visited, center):
    idxs = kdtree.query_ball_point(center, 5)
    for voxel in kdtree.data[idxs]:
        visited.add(tuple([int(v) for v in voxel]))


def gaussian_3d(xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, amplitude, offset):
    x, y, z = xyz
    value = (
        amplitude
        * np.exp(
            -(
                ((x - x0) ** 2) / (2 * sigma_x**2)
                + ((y - y0) ** 2) / (2 * sigma_y**2)
                + ((z - z0) ** 2) / (2 * sigma_z**2)
            )
        )
        + offset
    ).ravel()
    return value


def get_patch(img, voxel, shape, from_center=True):
    start, end = img_util.get_start_end(voxel, shape, from_center=from_center)
    return img[0, 0, start[2]: end[2], start[1]: end[1], start[0]: end[0]]


def is_inbounds(img, voxel, margin=16):
    """
    Check if a voxel is within bounds of a 3D image, with a specified margin.

    Parameters
    ----------
    img : ArrayLike
        The 3D image array.
    voxel : Tuple[int]
        The coordinates of the voxel to check (x, y, z).
    margin : int, optional
        Margin distance from the edges of the image. The default is 16.

    Returns
    -------
    bool
        True if the voxel is within bounds, considering the margin, and
        False otherwise.

    """
    for i in range(3):
        if voxel[i] < margin or voxel[i] >= img.shape[i] - margin:
            return False
    return True
