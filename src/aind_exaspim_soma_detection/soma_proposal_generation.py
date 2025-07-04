"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

    Soma Proposal Generation Algorithm
        1. Generate Initial Proposals - detect_blobs()
            a. Smooth image with Gaussian filter to reduce false positives.
            b. Laplacian of Gaussian (LoG) with multiple sigmas to enhance
               regions where the gradient changes rapidly, then apply non-
               linear maximum filter.
            c. Generate initial set of proposals by detecting local maximas.
            d. Shift each proposal to the brightest voxel in its neighborhood
               and reject it if the brightness is below a threshold.

        2. Filter Initial Proposals - filter_proposals()
            a. Merges proposals within a given distance threshold.
            b. If the number of proposals exceeds a certain threshold, the top
               k brightest proposals are kept.
            c. Fit Gaussian to neighborhood centered at proposal and compute
               fitness score by comparing fitted Gaussian to image values.
               Proposals are discarded if (1) fitness score is below threshold
               or (2) estimated standard deviation is out of range.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample
from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from tqdm import tqdm

import numpy as np

from aind_exaspim_soma_detection.utils import img_util
from aind_exaspim_soma_detection.utils.img_util import get_patch


# --- Wrappers ---
def generate_proposals(
    img_prefix,
    multiscale,
    patch_shape,
    patch_overlap,
    bright_threshold=0,
):
    """
    Generates somas proposals across a whole brain 3D image by dividing the
    image into patches. The coordinate of each image patch is assigned to a
    thread in order to optimize the runtime of this process.

    Parameters
    ----------
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    multiscale : int
        Level in the image pyramid that image patches are read from.
    patch_shape : Tuple[int]
        Shape of each image patch.
    patch_overlap : int
        Overlap between adjacent image patches in each dimension.
    bright_threshold : int, optional
        Brightness threshold used to filter proposals and image patches.
        Default is 0.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of proposals.
    """
    # Initializations
    img = img_util.open_img(img_prefix)
    margin = np.min(patch_overlap) // 4
    offsets = img_util.calculate_offsets(img, patch_shape, patch_overlap)

    # Generate proposals
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for offset in offsets:
            threads.append(
                executor.submit(
                    generate_proposals_patch,
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
        pbar = tqdm(total=len(threads), dynamic_ncols=True)
        for thread in as_completed(threads):
            proposals.extend(thread.result())
            pbar.update(1)
        pbar.update(1)
    return spatial_filtering(proposals, 60)


def generate_proposals_patch(
    img,
    offset,
    margin,
    patch_shape,
    multiscale,
    bright_threshold=150,
):
    """
    Generates soma proposals by detecting blobs, filtering them, and
    converting the proposal coordinates from image to physical space.

    Parameters
    ----------
    img : zarr.core.Array
        Array representing a 3D image of a whole brain.
    offset : Tuple[int]
        The offset of the image patch to be extracted from "img". Note that
        proposals will be generated within this image patch.
    margin : int
        Margin distance from the edges of the image that is used to filter
        blobs.
    patch_shape : List[int]
        Shape of the image patch to be extracted from "img".
    multiscale : int
        Level in the image pyramid that image patches are read from.
    bright_threshold : int, optional
        Minimum brightness required for image patch. Default is 150.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of proposals.
    """
    # Get image patch
    img_patch = get_patch(img, offset, patch_shape, from_center=False)
    if np.max(img_patch) < bright_threshold:
        return list()

    # Generate initial proposals
    img_patch = gaussian_filter(img_patch, sigma=0.5)
    proposals_1 = detect_blobs(img_patch, bright_threshold, 8, margin)
    proposals_2 = detect_blobs(img_patch, bright_threshold, 5, margin)
    proposals_3 = detect_blobs(img_patch, bright_threshold, 3.25, margin)
    proposals = proposals_1 + proposals_2 + proposals_3

    # Filter initial proposals + convert coordinates
    filtered_proposals = list()
    for voxel in filter_proposals(img_patch, proposals):
        filtered_proposals.append(
            img_util.local_to_physical(voxel, offset, multiscale)
        )
    return filtered_proposals


# -- Step 1: Generate Initial Proposals ---
def detect_blobs(img_patch, bright_threshold, LoG_sigma, margin):
    """
    Detects blob-like structures in a given image patch using Laplacian of
    Gaussian (LoG) method and filters them based on brightness and location.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image patch that blobs are to be detected in.
    bright_threshold : float
        Minimum brightness required for detected blobs.
    LoG_sigma : float
        Standard deviation of the Gaussian kernel for the LoG operation.
    margin : int
        Margin distance from the edges of the image used to filter blobs.

    Returns
    -------
    List[Tuple[int]]
        Voxel coordinates of detected blobs.
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
    Shifts each proposal to the brightest voxel in neighborhood.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image patch.
    proposals : List[Tuple[int]]
        Voxel coordinates of proposals.
    bright_threshold : int
        Minimum brightness required for each proposal.
    d : int, optional
        Size of neighborhood in each dimension. Default is 5.

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
        A 3D image patch containing the given proposals.
    voxel : Tuple[int]
        Center coordinate of the neighborhood.
    d : int
        Size of neighborhood in each dimension.

    Returns
    -------
    Tuple[int]
        Coordinate of bright voxel in neighborhood.
    """
    # Initializations
    i, j, k = voxel
    r = d // 2

    # Neighborhood bounds
    i_min, i_max = max(i - r, 0), min(i + r + 1, img_patch.shape[0])
    j_min, j_max = max(j - r, 0), min(j + r + 1, img_patch.shape[1])
    k_min, k_max = max(k - r, 0), min(k + r + 1, img_patch.shape[2])

    # Find the argmax
    nbhd = img_patch[i_min:i_max, j_min:j_max, k_min:k_max]
    argmax = np.unravel_index(np.argmax(nbhd), nbhd.shape)
    return (argmax[0] + i_min, argmax[1] + j_min, argmax[2] + k_min)


# --- Step 2: Filter Initial Proposals ---
def filter_proposals(img_patch, proposals, max_proposals=10, radius=5):
    """
    Filters a list of proposals based on multiple criteria including distance,
    brightness, and Gaussian fitness.

    Parameters
    ----------
    img_patch : np.ndarray
        A 3D image patch containing the given proposals.
    proposals : List[Tuple[int]]
        Voxel coordinates of proposals.
    max_proposals : int, optional
        Maximum number of proposals to return. Default is 10.
    radius : int, optional
        Radius (in voxels) to use for spatial filtering. Default is 5.

    Returns
    -------
    List[Tuple[float]]
        Filtered list of proposals.
    """
    # Filter by distance and brightness
    proposals = spatial_filtering(proposals, radius)
    if len(proposals) > max_proposals:
        proposals = brightness_filtering(img_patch, proposals, max_proposals)

    # Filter by Gaussian fitness
    proposals = gaussian_fit_filtering(img_patch, proposals)
    return proposals


def spatial_filtering(proposals, radius):
    """
    Filters a list of proposals by merging nearby proposals based on a given
    distance threshold.

    Parameters
    ----------
    proposals : List[Tuple[float]]
        Voxel coordinates of proposals.
    radius : float
        Distance that is used to find nearby proposals to be merged.

    Returns
    -------
    List[Tuple[float]]
        Filtered list of proposals.
    """
    filtered_proposals = list()
    if len(proposals) > 0:
        kdtree = KDTree(proposals)
        visited = set()
        for query in map(tuple, proposals):
            if query not in visited:
                # Search nbhd
                nbs = list()
                idxs = kdtree.query_ball_point(query, radius)
                for coord in map(tuple, kdtree.data[idxs]):
                    nbs.append(coord)
                    visited.add(coord)

                # Generate coordinate to add
                nbs = np.vstack(nbs)
                filtered_proposals.append(tuple(np.mean(nbs, axis=0)))
    return filtered_proposals


def brightness_filtering(img_patch, proposals, k):
    """
    Filters a list of proposals by keeping the top "k" brightest.

    Parameters
    ----------
    img_patch : np.ndarray
        A 3D image patch containing the given proposals.
    proposals : List[Tuple[int]]
        Voxel coordinates of proposals.
    k : int
        Maximum number of proposals to return.

    Returns
    -------
    List[Tuple[float]]
        Filtered list of proposals.
    """
    brightness = list()
    for proposal in proposals:
        proposal = tuple(map(int, proposal))
        brightness.append(img_patch[proposal])
    brightest_idxs = np.argsort(brightness)[::-1]
    return [proposals[idx] for idx in brightest_idxs[:k]]


def gaussian_fit_filtering(img_patch, proposals, r=4, min_score=0.7):
    """
    Filters a list of proposals by fitting a gaussian to neighborhood of each
    proposal and then checking the closeness of the fit.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image patch containing the given proposals.
    proposals : List[Tuple[int]]
        Voxel coordinates of proposals.
    r : int, optional
        Shape of neighborhood centered at each proposal that Gaussian is
        fitted to. Default is 4.
    min_score : float, optional
        Minimum fitness score that is used to filter proposals, which must be
        a value between 0 and 1. Default is 0.7.

    Returns
    -------
    List[Tuple[int]]
        Filtered and adjusted list of proposals.
    """
    filtered_proposals = list()
    for proposal in proposals:
        # Extract neighborhood
        x0, y0, z0 = tuple(map(int, proposal))
        x_min, x_max = max(0, x0 - r), min(img_patch.shape[0], x0 + r + 1)
        y_min, y_max = max(0, y0 - r), min(img_patch.shape[1], y0 + r + 1)
        z_min, z_max = max(0, z0 - r), min(img_patch.shape[2], z0 + r + 1)
        subpatch = img_patch[x_min:x_max, y_min:y_max, z_min:z_max]

        # Fit Gaussian
        params, voxels = img_util.fit_gaussian_3d(subpatch)
        score = img_util.compute_fit_score(subpatch, params, voxels)
        mean, std = params[0:3], abs(params[3:6])

        # Check whether to keep proposal
        feasible_range = all(std > 0.4) and all(std < 10)
        if score > min_score and (feasible_range and np.mean(std) > 0.75):
            proposal = [proposal[i] + mean[i] - r for i in range(3)]
            if is_inbounds(img_patch.shape, proposal, 1):
                filtered_proposals.append(proposal)
    return filtered_proposals


# --- Helpers ---
def is_inbounds(shape, voxel, margin):
    """
    Check if voxel is within bounds of a 3D image, with a specified margin.

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
