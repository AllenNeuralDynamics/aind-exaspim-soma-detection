"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

"""

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from skimage.feature import peak_local_max

from aind_exaspim_soma_detection.utils import img_util

BRIGHT_THRESHOLD = 160
LOG_SIGMA = 5
LOG_THRESHOLD = 5


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
        return list()

    # Generate initial proposals
    centers = detect_blobs(
        img_patch, bright_threshold, LoG_sigma, LoG_threshold
    )

    # Filter candidates + convert coordinates
    physical_centers = list()
    for voxel in filter_centers(img_patch, centers, margin):
        physical_centers.append(
            img_util.local_to_physical(voxel[::-1], offset, downsample_factor)
        )
    return physical_centers


def detect_blobs(img_patch, bright_threshold, LoG_sigma, LoG_threshold):
    # Preprocess image
    smoothed = gaussian_filter(img_patch, sigma=0.5)
    LoG = gaussian_laplace(smoothed, LoG_sigma)
    max_LoG = maximum_filter(LoG, 6)

    # Detect local peaks
    peaks = list()
    for peak in peak_local_max(max_LoG, min_distance=6):
        peak = tuple([int(x) for x in peak])
        if LoG[peak] > LoG_threshold:
            peaks.append(peak)
    return adjust_centers_by_brightness(img_patch, peaks, bright_threshold)


# --- Postprocess Centers ---
def adjust_centers_by_brightness(img_patch, centers, bright_threshold, n=6):
    """
    Adjust centers in a 3D image to the location of the brightest voxel in a
    local neighborhood.

    Parameters
    ----------
    img_patch : np.ndarray
        A 3D image where voxel intensity values used to identify the brightest
        voxel in a neighborhood.
    centers : list of tuples
        A list of coordinates representing the initial centers to be adjusted.
        Each coordinate is a tuple of integers (x, y, z).
    bright_threshold : int
        ...
    n : int, optional
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
        try:
            voxel = tuple([int(c) for c in center])
            voxel = find_argmax_in_nbhd(img_patch, voxel, n)
            if img_patch[voxel] > bright_threshold:
                adjusted_centers.add(tuple(voxel))
        except ValueError:
            pass
    return list(adjusted_centers)


def find_argmax_in_nbhd(img_patch, voxel, n):
    """
    Finds the coordinate of the maximum value within an n x n x n neighborhood
    of a 3D image centered at the given coordinates.

    Parameters:
    -----------
    img_patch : numpy.ndarray
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
    i_min, i_max = max(i - half_n, 0), min(i + half_n + 1, img_patch.shape[0])
    j_min, j_max = max(j - half_n, 0), min(j + half_n + 1, img_patch.shape[1])
    k_min, k_max = max(k - half_n, 0), min(k + half_n + 1, img_patch.shape[2])

    # Find the argmax
    nbhd = img_patch[i_min:i_max, j_min:j_max, k_min:k_max]
    argmax = np.unravel_index(np.argmax(nbhd), nbhd.shape)
    return (argmax[0] + i_min, argmax[1] + j_min, argmax[2] + k_min)


# --- Filtering ---
def filter_centers(img_patch, centers, margin, radius=6):
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
            # Check whether to keep
            center = tuple([int(v) for v in centers[idx]])
            fit, params = gaussian_fitness(img_patch, center, radius=radius)
            feasible_params = all(params[3:6] > 0.55) and all(params[3:6] < 10)
            if fit > 0.8 and feasible_params:
                center = [int(center[i] + params[i] - radius) for i in range(3)]
                filtered_centers.append(tuple(center))
                discard_nearby_centers(kdtree, visited, center)
    return filtered_centers


def gaussian_fitness(img, voxel, radius):
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
    patch = img[x_min:x_max, y_min:y_max, z_min:z_max]
    img_vals = patch.ravel()

    # Generate coordinates
    x = np.linspace(0, patch.shape[0], patch.shape[0])
    y = np.linspace(0, patch.shape[1], patch.shape[1])
    z = np.linspace(0, patch.shape[2], patch.shape[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    coords = (x.ravel(), y.ravel(), z.ravel())

    # Fit Gaussian
    try:
        amplitude = np.max(patch)
        offset = np.min(patch)
        shape = patch.shape
        x0, y0, z0 = shape[0] // 2, shape[1] // 2, shape[2] // 2
        p0 = (x0, y0, z0, 2, 2, 2, amplitude, offset)
        params, _ = curve_fit(gaussian_3d, coords, img_vals, p0=p0)
        return fitness_quality(img_vals, coords, params), params
    except RuntimeError:
        return 0, np.zeros((9))


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
    filtered_xyz_list = list()
    kdtree = KDTree(xyz_list)
    visited = set()
    for xyz_query in map(tuple, xyz_list):
        if xyz_query not in visited:
            # Search nbhd
            points = list()
            idxs = kdtree.query_ball_point(xyz_query, 20)
            for xyz in map(tuple, kdtree.data[idxs]):
                points.append(xyz)
                visited.add(xyz)

            # Generate point to add
            points = np.vstack(points)
            filtered_xyz_list.append(tuple(np.mean(points, axis=0)))
    return filtered_xyz_list


# --- utils ---
def discard_nearby_centers(kdtree, visited, center):
    idxs = kdtree.query_ball_point(center, 8)
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
