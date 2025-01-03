"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

    Soma Proposal Generation Algorithm
        1. Detect Initial Proposals - detect_blobs()
            a. Smooth image with Gaussian filter to reduce false positives.
            b. Laplacian of Gaussian (LoG) to enhance regions where the
               intensity changes dramatically (i.e. higher gradient), then
               apply a non-linear maximum filter.
            c. Generate initial set of proposals by detecting local maximas
               that lie outside of the image margins.
            d. Shift each proposal to the brightest voxel in its neighborhood.
               If the brightness is below a threshold, reject the proposal.

        2. Filter Initial Proposals - filter_proposals()
            a. Fit Gaussian to neighborhood centered at proposal.
            b. Compute closeness of fit by comparing fitted Gaussian and image
               values.
            c. Proposals are discarded if (1) fitness is below a threshold or
               (2) standard deviation of Gaussian is out of range.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from tqdm import tqdm

import numpy as np

from aind_exaspim_soma_detection.utils import img_util
from aind_exaspim_soma_detection.utils.img_util import get_patch

# Default parameters
BRIGHT_THRESHOLD = 160


# --- Wrappers ---
def run_on_whole_brain(
    img_prefix,
    overlap,
    patch_shape,
    multiscale,
    bright_threshold=BRIGHT_THRESHOLD,
):
    # Initializations
    img = img_util.open_img(img_prefix)
    margin = np.min(overlap) // 4
    offsets = img_util.sliding_window_coords_3d(img, patch_shape, overlap)

    # Generate proposals
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for offset in offsets:
            threads.append(
                executor.submit(
                    generate_proposals,
                    img,
                    offset,
                    margin,
                    patch_shape,
                    multiscale,
                    bright_threshold,
                )
            )

        # Process thread
        proposals = list()
        pbar = tqdm(total=len(threads))
        for thread in as_completed(threads):
            proposals.extend(thread.result())
            pbar.update(1)
        pbar.update(1)
    return global_filtering(proposals)


def generate_proposals(
    img,
    offset,
    margin,
    patch_shape,
    multiscale,
    bright_threshold=BRIGHT_THRESHOLD,
):
    """
    Generates soma proposals by detecting blobs, filtering them, and
    converting the proposal coordinates from image to physical space.

    Parameters
    ----------
    img : zarr.core.Array
        A 3D array representing an image of a whole brain.
    offset : Tuple[int]
        The offset of the image patch to be extracted from "img". Note that
        proposals will be generated within this image patch.
    margin : int
        Margin distance from the edges of the image that is used to filter
        blobs.
    patch_shape : List[int]
        Shape of the image patch to be extracted from "img".
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.
    bright_threshold : int, optional
        Minimum brightness required for image patch. the default is the global
        variable "BRIGHT_THRESHOLD".

    Returns
    -------
    List[Tuple[int]]
        List of physical coordinates of proposals.

    """
    # Get image patch
    img_patch = get_patch(img, offset, patch_shape, from_center=False)
    if np.max(img_patch) < bright_threshold:
        return list()

    # Generate initial proposals
    img_patch = gaussian_filter(img_patch, sigma=0.5)
    proposals_1 = detect_blobs(img_patch, bright_threshold, 5, margin)
    proposals_2 = detect_blobs(img_patch, bright_threshold, 3.5, margin)
    proposals = proposals_1 + proposals_2

    # Filter initial proposals + convert coordinates
    filtered_proposals = list()
    for voxel in filter_proposals(img_patch, proposals):
        filtered_proposals.append(
            img_util.local_to_physical(voxel[::-1], offset, multiscale)
        )
    return filtered_proposals


# -- Step 1: Detect Initial Proposals ---
def detect_blobs(img_patch, bright_threshold, LoG_sigma, margin):
    """
    Detects blob-like structures in a given image patch using Laplacian of
    Gaussian (LoG) method and filters them based on brightness and location.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image that blobs are to be detected in.
    bright_threshold : float
        Minimum brightness required for detected blobs.
    LoG_sigma : float
        Standard deviation of the Gaussian kernel for the LoG operation.
    margin : int
        Margin distance from the edges of the image that is used to filter
        blobs.

    Returns
    -------
    List[Tuple[int]]
        List of voxel coordinates of detected blobs.

    """
    blobs = list()
    LoG = gaussian_laplace(img_patch, LoG_sigma)
    for peak in peak_local_max(maximum_filter(LoG, 5), min_distance=5):
        peak = tuple([int(x) for x in peak])
        if LoG[peak] > 0 and is_inbounds(img_patch.shape, peak, margin):
            blobs.append(peak)
    return shift_to_brightest(img_patch, blobs, bright_threshold)


def shift_to_brightest(img_patch, proposals, bright_threshold, d=5):
    """
    Shifts each proposal to the brightest voxel in a local neighborhood.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image where intensity values are used to identify the brightest
        voxel in a neighborhood.
    proposals : List[Tuple[int]]
        List of voxel coordinates of proposals to be shifted.
    bright_threshold : int
        Minimum brightness required for each proposal.
    d : int, optional
        Size of the neighborhood in each dimension. The default is 6.

    Returns
    -------
    List[Tuple[int]]
        Shifted proposals.

    """
    shifted_proposals = set()
    for proposal in proposals:
        proposal = tuple([int(p) for p in proposal])
        voxel = find_argmax_in_nbhd(img_patch, proposal, d)
        if img_patch[voxel] > bright_threshold:
            shifted_proposals.add(voxel)
    return list(shifted_proposals)


def find_argmax_in_nbhd(img_patch, voxel, d):
    """
    Finds the brightest voxel in a d x d x d neighborhood centered at the
    given voxel coordinate.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D array representing an image.
    voxel : Tuple[int]
        Center coordinate of the neighborhood.
    d : int
        Size of the neighborhood in each dimension.

    Returns
    -------
    Tuple[int]
        Coordinate of bright voxel in neighborhood.

    """
    # Initializations
    i, j, k = voxel
    half_d = d // 2

    # Neighborhood bounds
    i_min, i_max = max(i - half_d, 0), min(i + half_d + 1, img_patch.shape[0])
    j_min, j_max = max(j - half_d, 0), min(j + half_d + 1, img_patch.shape[1])
    k_min, k_max = max(k - half_d, 0), min(k + half_d + 1, img_patch.shape[2])

    # Find the argmax
    nbhd = img_patch[i_min:i_max, j_min:j_max, k_min:k_max]
    argmax = np.unravel_index(np.argmax(nbhd), nbhd.shape)
    return (argmax[0] + i_min, argmax[1] + j_min, argmax[2] + k_min)


# --- Step 2: Filter Initial Proposals ---
def filter_proposals(img_patch, proposals, radius=5):
    """
    Filters a list of proposals by fitting a gaussian to neighborhood of each
    proposal and then checking the closeness of the fit.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image patch containing proposals.
    proposals : List[Tuple[int]]
        List of voxel coordinates of the proposals to be filtered.
    radius : int, optional
        Shape of neighborhood centered at each proposal that Gaussian is
        fitted to. The default is 5.

    Returns
    -------
    List[Tuple[int]]
        Filtered and adjusted proposals.

    """
    filtered_proposals = list()
    for proposal in proposals:
        # Fit Gaussian
        proposal = tuple(map(int, proposal))
        fit, params = gaussian_fitness(img_patch, proposal, radius)
        mean, std = params[0:3], abs(params[3:6])

        # Check whether to filter
        feasible_range = all(std > 0.4) and all(std < 10)
        if fit > 0.75 and (feasible_range and np.mean(std) > 0.65):
            proposal = [proposal[i] + mean[i] - radius for i in range(3)]
            filtered_proposals.append(proposal)
    return filtered_proposals


def gaussian_fitness(img_patch, voxel, r):
    """
    Fits a 3D Gaussian to neighborhood centered at "voxel" and computes the
    closeness of the fit.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image containing neighborhood that Gaussian is to be fitted to.
    voxel : Tuple[int]
        Voxel coordinate specifying the center of the neighborhood.
    r : int
        Radius of neighborhood around the center voxel. The neighborhood size
        is (2 * r + 1) ** 3.

    Returns
    -------
    tuple
        Fitness score and parameters of fitted Gaussian.

    """
    # Extract neighborhood
    x0, y0, z0 = voxel
    x_min, x_max = max(0, x0 - r), min(img_patch.shape[0], x0 + r + 1)
    y_min, y_max = max(0, y0 - r), min(img_patch.shape[1], y0 + r + 1)
    z_min, z_max = max(0, z0 - r), min(img_patch.shape[2], z0 + r + 1)
    nbhd = img_patch[x_min:x_max, y_min:y_max, z_min:z_max]
    img_vals = nbhd.ravel()

    # Generate coordinates
    xyz = [np.linspace(0, nbhd.shape[i], nbhd.shape[i]) for i in range(3)]
    x, y, z = np.meshgrid(xyz[0], xyz[1], xyz[2], indexing="ij")
    xyz = (x.ravel(), y.ravel(), z.ravel())

    # Fit Gaussian
    try:
        # Initialize guess for parameters
        amplitude = np.max(nbhd)
        offset = np.min(nbhd)
        shape = nbhd.shape
        x0, y0, z0 = shape[0] // 2, shape[1] // 2, shape[2] // 2
        p0 = (x0, y0, z0, 2, 2, 2, amplitude, offset)

        # Fit
        params, _ = curve_fit(gaussian_3d, xyz, img_vals, p0=p0)
        fitness_score = fitness_quality(img_vals, xyz, params)
        return fitness_score, params
    except RuntimeError:
        return 0, np.zeros((9))


def fitness_quality(img_patch, voxels, params):
    """
    Evaluates the quality of a fitten Gaussian by computing the correlation
    coefficient between the image values and fitted Gaussian values.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D array representing an image.
    voxels : Tuple[numpy.ndarray]
        Flattened arrays of voxel coordinates.
    params : numpy.ndarray
        Parameters of the fitted Gaussian function.

    Returns
    -------
    float
        Correlation coefficient between the image values and fitted Gaussian
        values.

    """
    fitted_gaussian = gaussian_3d(voxels, *params).reshape(img_patch.shape)
    fitted = fitted_gaussian.flatten()
    actual = img_patch.flatten()
    return np.corrcoef(actual, fitted)[0, 1]


def global_filtering(xyz_list):
    """
    Filters a list of 3D points by merging nearby points based on a given
    distance threshold.

    Parameters
    ----------
    xyz_list : List[Tuple[float]]
        List of xyz coordinates.

    Returns
    -------
    List[Tuple[float]]
        Filtered list of the given 3D points.

    """
    filtered_xyz_list = list()
    kdtree = KDTree(xyz_list)
    visited = set()
    for xyz_query in map(tuple, xyz_list):
        if xyz_query not in visited:
            # Search nbhd
            points = list()
            idxs = kdtree.query_ball_point(xyz_query, 24)
            for xyz in map(tuple, kdtree.data[idxs]):
                points.append(xyz)
                visited.add(xyz)

            # Generate point to add
            points = np.vstack(points)
            filtered_xyz_list.append(tuple(np.mean(points, axis=0)))
    return filtered_xyz_list


# --- utils ---
def gaussian_3d(xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, amplitude, offset):
    """
    Computes the values of a 3D Gaussian at the given coordinates.

    Parameters
    ----------
    xyz : Tuple[ArrayLike]
        Coordinates that Gaussian is to be evaluated.
    x0, y0, z0 : float
        Mean of Gaussian.
    sigma_x, sigma_y, sigma_z : float
        Standard deviations of Gaussian.
    amplitude : float
        Peak value (amplitude) of Gaussian at the center.
    offset : float
        Constant value added to Gaussian that represents the baseline offset.

    Returns
    -------
    numpy.ndarray
        Computed values of the 3D Gaussian at the given coordinates. Note that
        these values are flattened from a 3D grid to a 1D array.

    """
    x, y, z = xyz
    value = offset + amplitude * np.exp(
        -(
            ((x - x0) ** 2) / (2 * sigma_x**2)
            + ((y - y0) ** 2) / (2 * sigma_y**2)
            + ((z - z0) ** 2) / (2 * sigma_z**2)
        )
    )
    return value.ravel()


def is_inbounds(shape, voxel, margin):
    """
    Check if a voxel is within bounds of a 3D image, with a specified margin.

    Parameters
    ----------
    shape : ArrayLike
        Shape of the 3D image.
    voxel : Tuple[int]
        Voxel coordinate to be checked.
    margin : int
        Margin distance from the edges of the image.

    Returns
    -------
    bool
        True if the voxel is outside of margins, and False otherwise.

    """
    for i in range(3):
        if voxel[i] < margin or voxel[i] > shape[i] - margin:
            return False
    return True
