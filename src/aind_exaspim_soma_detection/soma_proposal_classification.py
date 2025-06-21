"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals from a whole-brain with a convolutional
neural network.

"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from scipy.spatial.distance import euclidean
from scipy.optimize import curve_fit
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from aind_exaspim_soma_detection.utils import img_util, ml_util
from aind_exaspim_soma_detection.machine_learning.data_handling import (
    MultiThreadedDataLoader,
    ProposalDataset,
)


def classify_proposals(
    brain_id,
    proposals,
    img_path,
    model_path,
    multiscale,
    patch_shape,
    threshold,
    batch_size=64,
    device="cuda",
):
    """
    Classifies soma proposals using a pre-trained model and returns the
    physical coordinates of accepted soma proposals.

    Parameters
    ----------
    brain_id : str
        Unique identifier for the whole-brain dataset.
    proposals : List[Tuple[float]]
        List of proposals, where each is represented by an xyz coordinate.
    img_path : str
        Path to whole-brain image stored in a S3 bucket.
    model_path : str
        Path to the pre-trained model that is used to classify the proposals.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.
    patch_shape : tuple of int
        Shape of image patches to be used for inference.
    threshold : float
        Threshold above which a proposal is called a soma.
    batch_size : int, optional
        Batch size used to run inference on the proposals. The default is 16.
    device : str, optional
        Name of device where model should be loaded and run. The default is
        "cuda".

    Returns:
    --------
    List[Tuple[float]]
        Physical coordinates of somas detected by the model.
    """
    # Initialize dataset
    proposals = [img_util.to_voxels(p, multiscale) for p in proposals]
    dataset = ProposalDataset(patch_shape)
    dataset.ingest_proposals(brain_id, img_path, proposals)

    # Generate predictions
    dataloader = MultiThreadedDataLoader(dataset, batch_size)
    model = ml_util.load_model(model_path, patch_shape, device)
    id_voxel, hat_y, _ = run_inference(dataloader, model, device)

    # Extract predicted somas
    soma_xyz_list = list()
    for (_, voxel), hat_y_i in zip(id_voxel, hat_y):
        if hat_y_i > threshold:
            soma_xyz_list.append(img_util.to_physical(voxel, multiscale))
    return soma_xyz_list


def run_inference(dataloader, model, device="cuda", verbose=True):
    """
    Runs inference on a given dataset using the provided model, then returns
    the predictions and ground truth labels.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader object that loads the dataset in batches.
    model : torch.nn.Module
        Neural network model that is used to generate predictions.
    device : str, optional
        Name of device where model should be loaded and run. The default is
        "cuda".
    verbose : bool, optional
        Indication of whether to display a progress bar during inference. The
        default is True.

    Returns
    -------
    Tuple[list]
        Tuple that contains the following:
            - "id_voxel" (List[tuple]): Unique identifier for each proposal
            that consists of the "brain_id" and "voxel" coordinate.
            - "hat_y" (numpy.ndarray): Prediction for each proposal.
            - "y" (numpy.ndarray): Ground truth label for each proposal.
    """
    # Initializations
    n = dataloader.n_rounds
    iterator = tqdm(dataloader, total=n) if verbose else dataloader

    # Main
    id_voxel, hat_y, y = list(), list(), list()
    with torch.no_grad():
        model.eval()
        for id_voxel_i, x_i, y_i in iterator:
            # Forward pass
            x_i = x_i.to(device)
            hat_y_i = torch.sigmoid(model(x_i))

            # Store result
            id_voxel.extend(id_voxel_i)
            hat_y.append(ml_util.toCPU(hat_y_i))
            y.append(np.array(y_i) if y_i[0] is not None else list())
    return id_voxel, np.vstack(hat_y)[:, 0], np.vstack(y)[:, 0]


# --- Accepted Proposal Filtering ---
def compute_metrics(
    img_path,
    accepts,
    multiscale,
    patch_shape,
    batch_size=64,
    min_branch_dist=120,
    min_brightness=250,
):
    """
    Filters a list of accepted proposals by checking whether there exists a
    branch emanating from the center, see "branch_filtering" for details.

    Parameters
    ----------
    img_path : str
        Path to whole-brain image stored in a S3 bucket.
    accepts : List[Tuple[float]]
        List of accepted proposals, where each is represented by an xyz
        coordinate.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.
    patch_shape : Tuple[int]
        Shape of the image patches to be read in order to check for branches.

    Returns
    -------
    pd.DataFrame
        Data frame containing metrics computed for each detected soma.
    """
    def load_patch(voxel):
        patch = img_util.get_patch(img, voxel, patch_shape)
        return (voxel, patch)

    # Initializations
    pbar = tqdm(total=len(accepts))
    img = img_util.open_img(img_path)
    voxels = [img_util.to_voxels(p, multiscale) for p in accepts]

    # Main
    results = []
    for batch_voxels in generate_batches(voxels, batch_size):
        # Phase 1: Read patches concurrently
        with ThreadPoolExecutor() as exec:
            voxel_patches = list(exec.map(load_patch, batch_voxels))

        # Phase 2: Analyze patches concurrently
        process_patch_with_arg = partial(process_patch, multiscale=multiscale)
        with ProcessPoolExecutor() as exec:
            for result in exec.map(process_patch_with_arg, voxel_patches):
                if result:
                    is_branchy = result["Max_Branch_Dist"] > min_branch_dist
                    is_bright = result["Brightness"] > min_brightness
                    if is_branchy and is_bright:
                        results.append(result)
                pbar.update(1)
    return pd.DataFrame(results)


def process_patch(voxel_patch, multiscale=3):
    try:
        # Fit gaussian
        voxel, img_patch = voxel_patch
        params, voxels = fit_rotated_gaussian(img_patch)
        radii = compute_radii(params, multiscale)
        if (radii > 150).any():
            return None

        # Compute metrics
        result = {
            "xyz": img_util.to_physical(voxel, multiscale=multiscale),
            "Brightness": compute_soma_brightness(img_patch, params, voxels),
            "Volume (µm³)": int(np.prod(radii) * (4 / 3) * np.pi),
            "Radii (μm)": tuple([round(r, 2) for r in radii]),
            "Max_Branch_Dist": compute_branch_dist(img_patch, multiscale),
        }
        return result
    except Exception as e:
        print(f"[ERROR] Voxel {voxel} failed with error: {e}")
        return None


def compute_branch_dist(img_patch, multiscale):
    # Compute foreground
    relabeled = img_util.segment_3class_otsu(img_patch)
    object_mask = np.zeros_like(img_patch)

    root = tuple([s // 2 for s in img_patch.shape])
    root_xyz = img_util.to_physical(root, multiscale=multiscale)

    # Search center object
    max_dist = 0
    queue = [root]
    visited = set(queue)
    while len(queue) > 0:
        # Visit voxel
        voxel = queue.pop()
        xyz = img_util.to_physical(voxel, multiscale=multiscale)
        max_dist = max(euclidean(xyz, root_xyz), max_dist)
        object_mask[voxel] = 1

        # Update queue
        for nb in img_util.get_nbs(voxel, img_patch.shape):
            if nb not in visited and relabeled[nb]:
                queue.append(nb)
                visited.add(nb)
    return round(max_dist, 2)


def compute_soma_brightness(img_patch, params, voxels):
    mask = gaussian_mask(voxels, *params[:9]).reshape(img_patch.shape)
    return int(np.percentile(img_patch[mask], 80)) if mask.any() else 0


def compute_radii(params, multiscale, anisotropy=(0.748, 0.748, 1.0)):
    # Compute precision matrix
    scaled_anisotropy = [a * 2**multiscale for a in anisotropy]
    _, _, _, a11, a12, a13, a22, a23, a33, _, _ = params
    P_voxel = np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33]
    ])

    # Convert precision matrix from voxel to physical space
    S = np.diag(scaled_anisotropy)
    S_inv = np.linalg.inv(S)

    # Adjusted precision matrix in physical units
    P_physical = S_inv.T @ P_voxel @ S_inv
    try:
        cov_physical = np.linalg.inv(P_physical)
        eigvals = np.linalg.eigvalsh(cov_physical)
        radii = 2 * np.sqrt(np.abs(eigvals))
        return radii
    except np.linalg.LinAlgError:
        return np.zeros([0, 0, 0])


# --- Helpers ---
def fit_rotated_gaussian(img_patch):
    # Generate voxel coordinates
    center = [dim // 2 for dim in img_patch.shape]
    grid = np.meshgrid(
        np.arange(img_patch.shape[0]),
        np.arange(img_patch.shape[1]),
        np.arange(img_patch.shape[2]),
        indexing='ij'
    )
    voxels = np.stack(grid, axis=-1).reshape(-1, 3)

    # Initial guess for Gaussian parameters
    initial_guess = [
        center[0], center[1], center[2],
        1e-2, 0, 0,
        1e-2, 0,
        1e-2,
        np.max(img_patch), np.min(img_patch)
    ]

    # Fit rotated 3D Gaussian
    try:
        params, _ = curve_fit(
            gaussian_3d_rotated,
            voxels,
            img_patch.ravel(),
            p0=initial_guess
        )
    except RuntimeError:
        params = np.zeros(len(initial_guess))
    return params, voxels


def gaussian_3d_rotated(
    coords, x0, y0, z0, a11, a12, a13, a22, a23, a33, A, B
):
    # Refactor coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Construct quadratic form
    quad = (
        a11*dx**2 + 2*a12*dx*dy + 2*a13*dx*dz +
        a22*dy**2 + 2*a23*dy*dz + a33*dz**2
    )
    return A * np.exp(-0.5 * quad) + B


def gaussian_mask(
    coords, x0, y0, z0, a11, a12, a13, a22, a23, a33, threshold=4.0
):
    """
    Computes a binary mask of voxels within a specified Mahalanobis distance
    (default: 2 standard deviations => threshold=4) from the Gaussian center.

    Parameters
    ----------
    coords : ndarray of shape (N, 3)
        Voxel coordinates.
    x0, y0, z0 : float
        Center of the Gaussian.
    a11, a12, a13, a22, a23, a33 : float
        Elements of the symmetric positive-definite matrix defining the
        quadratic form. This matrix is the inverse of the covariance matrix.
    threshold : float
        Mahalanobis distance squared (e.g., 4.0 for 2 standard deviations).

    Returns
    -------
    mask : ndarray of shape (N,)
        Boolean array where True indicates the voxel is within the threshold.
    """

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - x0
    dy = y - y0
    dz = z - z0
    quad = (
        a11*dx**2 + 2*a12*dx*dy + 2*a13*dx*dz +
        a22*dy**2 + 2*a23*dy*dz + a33*dz**2
    )
    return quad <= threshold


def generate_batches(iterable, batch_size):
    """
    Yield successive batches from iterable.
    """
    for i in range(0, len(iterable), batch_size):
        n = min(i + batch_size, len(iterable) - 1)
        yield iterable[i:n]


def reformat_coords(coords):
    coord_list = list()
    for i in range(len(coords[0])):
        coord_list.append((coords[0][i], coords[1][i], coords[2][i]))
    return np.array(coord_list)
