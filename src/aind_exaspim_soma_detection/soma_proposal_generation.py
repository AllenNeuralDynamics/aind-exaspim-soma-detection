"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

    Soma Proposal Generation Algorithm
        1. Generate Initial Proposals - detect_blobs()
            a. Smooth image with Gaussian filter to reduce false positives.
            b. Laplacian of Gaussian (LoG) with multiple sigmas to enhance
               regions where the gradient changes rapidly.
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
from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from tqdm import tqdm

import numpy as np

from aind_exaspim_soma_detection.utils import img_util


# --- Wrappers ---
def generate_proposals(
    img_path,
    multiscale,
    patch_shape,
    patch_overlap,
    min_brightness=0,
):
    """
    Generates somas proposals across a whole brain 3D image by dividing the
    image into patches. The coordinate of each image patch is assigned to a
    thread in order to optimize the runtime of this process.

    Parameters
    ----------
    img_path : str
        Path to image.
    multiscale : int
        Level in the image pyramid that image patches are read from.
    patch_shape : Tuple[int]
        Shape of each image patch.
    patch_overlap : int
        Overlap between adjacent image patches in each dimension.
    min_brightness : int, optional
        Brightness threshold used to filter proposals and image patches.
        Default is 0.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of proposals.
    """
    img = img_util.TensorStoreImage(img_path)
    margin = np.min(patch_overlap) // 4
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for offset in img.generate_offsets(patch_shape, patch_overlap):
            threads.append(
                executor.submit(
                    generate_proposals_patch,
                    img,
                    offset,
                    margin,
                    patch_shape,
                    multiscale,
                    min_brightness,
                )
            )

        # Process thread
        proposals = list()
        pbar = tqdm(total=len(threads), dynamic_ncols=True)
        for thread in as_completed(threads):
            proposals.extend(thread.result())
            pbar.update(1)
    return spatial_filtering(proposals, 50)


def generate_proposals_patch(
    img,
    offset,
    margin,
    patch_shape,
    multiscale,
    min_brightness=0,
):
    """
    Generates soma proposals by detecting blobs and filters them by
    brightness and gaussian-like appearance.

    Parameters
    ----------
    img : TensorStoreImage
        3D image of a whole-brain.
    offset : Tuple[int]
        Offset of the image patch to extract from "img". Note that proposals
        will be generated within this patch.
    margin : int
        Margin distance from the edges of the image used to filter blobs.
    patch_shape : Tuple[int]
        Shape of the patch to be extracted.
    multiscale : int
        Level in the image pyramid that patches are read from.
    min_brightness : int, optional
        Minimum brightness required for image patch. Default is 0.

    Returns
    -------
    proposals : List[Tuple[float]]
        Physical coordinates of proposals.
    """
    # Get image patch
    img_patch = img.read(offset, patch_shape, is_center=False)
    if img_patch.max() < min_brightness:
        return list()

    img_patch = gaussian_filter(img_patch, sigma=1)

    # Generate initial proposals
    initial_proposals = list()
    for stdev in [3, 5, 8]:
        initial_proposals.extend(
            detect_blobs(img_patch, min_brightness, stdev, margin)
        )

    # Filter initial proposals + convert coordinates
    proposals = list()
    for voxel in filter_proposals(img_patch, initial_proposals):
        proposals.append(
            img_util.local_to_physical(voxel, offset, multiscale)
        )
    return proposals


# -- Step 1: Generate Initial Proposals ---
def detect_blobs(img, min_brightness, stdev, margin):
    """
    Detects blob-like structures in a given image patch using Laplacian of
    Gaussian (LoG) method and removes proposals in image margin.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D image patch that proposals are to be detected in.
    min_brightness : float
        Minimum brightness required for detected blobs.
    stdev : float
        Standard deviation of the Gaussian kernel for the LoG operation.
    margin : int
        Margin distance from the edges of the image used to filter blobs.

    Returns
    -------
    peaks : numpy.ndarray
        Voxel coordinates of detected blobs.
    """
    # Find blob-like objects
    LoG = gaussian_laplace(img, stdev)
    peaks = peak_local_max(LoG, min_distance=5)
    peaks = peaks[LoG[peaks[:, 0], peaks[:, 1], peaks[:, 2]] > 0]

    # Remove blob-like objects in image margin
    peaks = [p for p in peaks if img_util.is_inbounds(img.shape, p, margin)]
    peaks = shift_to_brightest(img, peaks, min_brightness)
    return peaks


def shift_to_brightest(img, proposals, min_brightness, d=5):
    """
    Shifts each proposal to the brightest voxel in its neighborhood.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D image patch.
    proposals : numpy.ndarray
        Voxel coordinates of proposals.
    min_brightness : int
        Minimum brightness required for each proposal.
    d : int, optional
        Size of neighborhood in each dimension. Default is 5.

    Returns
    -------
    numpy.ndarray
        Shifted proposals.
    """
    r = d // 2
    shifted = []
    for i, j, k in proposals:
        # Extract nbhd
        i_min, i_max = max(i - r, 0), min(i + r + 1, img.shape[0])
        j_min, j_max = max(j - r, 0), min(j + r + 1, img.shape[1])
        k_min, k_max = max(k - r, 0), min(k + r + 1, img.shape[2])
        nbhd = img[i_min:i_max, j_min:j_max, k_min:k_max]

        # Find brightest voxel and check threshold
        argmax = np.unravel_index(np.argmax(nbhd), nbhd.shape)
        voxel = (argmax[0] + i_min, argmax[1] + j_min, argmax[2] + k_min)
        if img[voxel] > min_brightness:
            shifted.append(voxel)
    return np.array(shifted) if shifted else np.empty((0, 3))


# --- Step 2: Filter Initial Proposals ---
def filter_proposals(img, proposals, max_proposals=10, radius=5):
    """
    Filters proposals by proximity to other proposals, distance,
    brightness, and Gaussian fitness.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D image patch containing the proposals.
    proposals : List[Tuple[int]]
        Voxel coordinates of proposals.
    max_proposals : int, optional
        Maximum number of proposals to return. Default is 10.
    radius : int, optional
        Radius (in voxels) to use for spatial filtering. Default is 5.

    Returns
    -------
    proposals : List[Tuple[float]]
        Filtered list of proposals.
    """
    # Filter by distance and brightness
    proposals = spatial_filtering(proposals, radius)
    if len(proposals) > max_proposals:
        brightness = img[proposals[:, 0], proposals[:, 1], proposals[:, 2]]
        idxs = np.argsort(brightness)[::-1]
        proposals = proposals[idxs[:max_proposals]]

    # Filter by Gaussian fitness
    proposals = gaussian_fit_filtering(img, proposals)
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
    filtered_proposals : numpy.ndarray
        Filtered list of proposals.
    """
    # Check for proposals
    if not proposals:
        return np.empty((0, 3), dtype=int)

    # Build KD-Tree of proposals
    proposals = np.array(proposals)
    kdtree = KDTree(proposals)

    # Search for duplicates
    filtered_proposals = []
    unvisited = set(range(len(proposals)))
    for i, query in enumerate(proposals):
        # Check if visited
        if i not in unvisited:
            continue

        # Search nbhd
        idxs = kdtree.query_ball_point(query, radius)
        unvisited -= set(idxs)
        centroid = proposals[idxs].mean(axis=0).round()
        filtered_proposals.append(centroid)
    return np.array(filtered_proposals, dtype=int)


def gaussian_fit_filtering(img, proposals, r=4, min_score=0.7):
    """
    Filters proposals by fitting a gaussian to neighborhood of each proposal
    and checks the closeness of the fit.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D image patch containing the given proposals.
    proposals : numpy.ndarray
        Voxel coordinates of proposals.
    r : int, optional
        Shape of neighborhood centered at each proposal that Gaussian is
        fitted to. Default is 4.
    min_score : float, optional
        Minimum fitness score that is used to filter proposals, which must be
        a value between 0 and 1. Default is 0.7.

    Returns
    -------
    final_proposals : List[Tuple[int]]
        Filtered and adjusted list of proposals.
    """
    final_proposals = list()
    for x0, y0, z0 in proposals:
        # Extract neighborhood
        x_min, x_max = max(0, x0 - r), min(img.shape[0], x0 + r + 1)
        y_min, y_max = max(0, y0 - r), min(img.shape[1], y0 + r + 1)
        z_min, z_max = max(0, z0 - r), min(img.shape[2], z0 + r + 1)
        subpatch = img[x_min:x_max, y_min:y_max, z_min:z_max]

        # Fit Gaussian
        params, voxels = img_util.fit_gaussian_3d(subpatch)
        score = img_util.compute_fit_score(subpatch, params, voxels)
        mean, std = params[0:3], abs(params[3:6])

        # Check whether to keep proposal
        feasible_range = all(std > 0.4) and all(std < 10)
        if score > min_score and (feasible_range and np.mean(std) > 0.75):
            proposal = np.array([x0, y0, z0]) + mean - r
            if img_util.is_inbounds(img.shape, proposal, margin=1):
                final_proposals.append(tuple(proposal.round()))
    return final_proposals
