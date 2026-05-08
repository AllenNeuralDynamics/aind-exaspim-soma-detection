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

from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from threading import Thread
from tqdm import tqdm

import numpy as np
import queue

from aind_exaspim_soma_detection.utils import img_util

_STOP = object()


def generate_proposals(
    img_path,
    multiscale,
    patch_shape,
    patch_overlap,
    min_brightness=0,
    n_loaders=16,
    max_queue_size=32,
    max_worker_processes=16,
):
    # Start loader threads
    margin = np.min(patch_overlap) // 4
    batches = create_batches(img_path, patch_shape, patch_overlap, n_loaders)
    patch_queue = queue.Queue(maxsize=max_queue_size)
    start_loaders(img_path, batches, patch_shape, min_brightness, patch_queue)

    # Drain queue → submit to bounded process pool
    all_proposals = []
    stops_seen = 0
    pending = []
    pbar = tqdm(total=np.sum([len(b) for b in batches]))
    with ProcessPoolExecutor(max_workers=max_worker_processes) as executor:
        while stops_seen < n_loaders:
            # Pop queue
            item = patch_queue.get()
            if item is _STOP:
                stops_seen += 1
                continue

            offset, img_patch = item

            # Pop completed process
            if len(pending) >= max_worker_processes * 2:
                done_future = pending.pop(0)
                all_proposals.extend(done_future.result())
                pbar.update(1)

            # Submit new process
            pending.append(
                executor.submit(
                    _process_patch,
                    offset,
                    img_patch,
                    margin,
                    multiscale,
                    min_brightness,
                )
            )

        # Drain remaining futures
        for future in pending:
            all_proposals.extend(future.result())
            pbar.update(1)

    return spatial_filtering(all_proposals, 50)


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
    pass


def start_loaders(
    img_path, offset_batches, shape, min_brightness, patch_queue
):
    """
    Spawns one loader thread per partition of offsets.

    Parameters
    ----------
    img_path : str
        Path to image to be read from.
    offset_batches : List[List[Tuple[int]]]
        Offset batches to be assigned to an image loader.
    shape : Tuple[int]
        Shape of image patches to be read.
    min_brightness : int
        Minimum brightness required for patch to be processed.
    patch_queue : Queue.queue
        Queue containing patches to be processed.
    """
    threads = []
    for offsets in offset_batches:
        t = Thread(
            target=_patch_loader,
            args=(img_path, offsets, shape, min_brightness, patch_queue),
            daemon=True,
        )
        t.start()
        threads.append(t)
    return threads


def _patch_loader(img_path, offsets, shape, min_brightness, patch_queue):
    """
    Reads image patches and places (offset, img_patch) onto a bounded queue.

    Parameters
    ----------
    img_path : str
        Path to image to be read from.
    offsets : List[Tuple[int]]
        Offsets of image patches to be read.
    shape : Tuple[int]
        Shape of image patches to be read.
    min_brightness : int
        Minimum brightness required for patch to be processed.
    patch_queue : Queue.queue
        Queue containing patches to be processed.
    """
    img = img_util.TensorStoreImage(img_path)
    for offset in offsets:
        # Read patch
        img_patch = img.read(offset, shape, is_center=False)

        # Check whether to place on queue
        if img_patch.max() > min_brightness:
            img_patch = gaussian_filter(img_patch, sigma=1)
            patch_queue.put((offset, img_patch))
    patch_queue.put(_STOP)


def _process_patch(offset, img_patch, margin, multiscale, min_brightness):
    """
    Pure function suitable for ProcessPoolExecutor.
    Receives an already-loaded, already-filtered patch.
    """
    # Detect blob-like objects
    proposals = []
    for stdev in [3, 5, 8]:
        proposals.extend(
            detect_blobs(img_patch, min_brightness, stdev, margin)
        )

    # Filter proposals
    proposals = filter_proposals(img_patch, proposals)
    if len(proposals) > 0:
        proposals += np.array(offset)
        proposals = [img_util.to_physical(v, multiscale) for v in proposals]
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
def filter_proposals(img, proposals, max_proposals=600, radius=5):
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
        Maximum number of proposals to return. Default is 640.
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
    filtered_proposals = list()
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
                filtered_proposals.append(tuple(proposal.round()))
    return np.array(filtered_proposals)


# --- Helpers ---
def create_batches(img_path, patch_shape, patch_overlap, n_loaders):
    """
    Creates offset batches to assign to image loaders.

    Parameters
    ---------
    img_path : str
        Path to image to be read from.
    patch_shape : Tuple[int]
        Shape of the patch to be extracted.
    patch_overlap : Tuple[int]
        Shape of overlap between adjacent patches.
    n_loaders : int
        Number of image loaders.

    Returns
    -------
    List[List[Tuple[int]]
        Offset batches to assign to image loaders.
    """
    img = img_util.TensorStoreImage(img_path)
    offsets = list(img.generate_offsets(patch_shape, patch_overlap))
    sz = max(1, len(offsets) // n_loaders)
    return [offsets[i: i + sz] for i in range(0, len(offsets), sz)]
