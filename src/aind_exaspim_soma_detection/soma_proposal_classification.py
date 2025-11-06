"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals from a whole-brain with a convolutional
neural network.

"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
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
    soma_xyz_list : List[Tuple[float]]
        Physical coordinates of somas detected by the model.
    """
    # Initialize dataset
    proposals = [img_util.to_voxels(p, multiscale) for p in proposals]
    dataset = ProposalDataset(patch_shape)
    dataset.ingest_proposals(brain_id, img_path, proposals)

    # Generate predictions
    dataloader = MultiThreadedDataLoader(dataset, batch_size)
    model = ml_util.load_model(model_path, patch_shape, device)
    id_voxel, hat_y = run_inference(dataloader, model, device)

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
        Name of device where model should be loaded and run. Default is
        "cuda".
    verbose : bool, optional
        Indication of whether to display a progress bar during inference.
        Default is True.

    Returns
    -------
    id_voxel : List[Tuple[str, Tuple[float]]]
        Unique identifier for each proposal that consists of the "brain_id"
        and "voxel" coordinate.
    hat_y : numpy.ndarray
        Prediction for each proposal.
    """
    # Initializations
    n = dataloader.n_rounds
    iterator = tqdm(dataloader, total=n) if verbose else dataloader

    # Main
    id_voxel, hat_y = list(), list()
    with torch.no_grad():
        model.eval()
        for id_voxel_i, x_i, _ in iterator:
            # Forward pass
            x_i = x_i.to(device)
            hat_y_i = torch.sigmoid(model(x_i))

            # Store result
            id_voxel.extend(id_voxel_i)
            hat_y.append(ml_util.toCPU(hat_y_i))

    # Reformat predictions
    hat_y = np.vstack(hat_y)[:, 0]
    return id_voxel, hat_y


# --- Accepted Proposal Filtering ---
def compute_metrics(
    img_path,
    accepts,
    multiscale,
    patch_shape,
    batch_size=64,
    min_brightness=200,
):
    """
    Filters a list of accepted proposals by checking whether there exists a
    branch emanating from the center, see "branch_filtering" for details.

    Parameters
    ----------
    img_path : str
        Path to whole-brain image stored in an S3 bucket.
    accepts : List[Tuple[float]]
        List of accepted proposals, where each is represented by an xyz
        coordinate.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.
    patch_shape : Tuple[int]
        Shape of the image patches to be read in order to check for branches.

    Returns
    -------
    results : pandas.DataFrame
        Dataframe containing metrics computed for each detected soma.
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
                    if result["Brightness"] > min_brightness:
                        results.append(result)
                pbar.update(1)
    return pd.DataFrame(results)


def process_patch(voxel_patch, multiscale=2):
    # Fit gaussian
    voxel, img_patch = voxel_patch
    params, voxels = img_util.fit_rotated_gaussian_3d(img_patch)
    mask = img_util.rotated_gaussian_3d_mask(
        img_patch.shape, voxels, *params[:9]
    )

    # Compute metrics
    brightness = np.percentile(img_patch[mask], 80) if mask.any() else 0
    radii = compute_radii(params, multiscale)
    score = img_util.compute_fit_score(img_patch, params, voxels)

    # Compile results
    feasible_radii = (radii > 6).any() or (radii < 140).any()
    if feasible_radii and score > 0.7:
        xyz = img_util.to_physical(voxel, multiscale=multiscale)
        result = {
            "xyz": tuple(round(float(t), 2) for t in xyz),
            "voxel": tuple(int(t) for t in voxel),
            "Brightness": int(brightness),
            "Volume (µm³)": int(np.prod(radii) * (4 / 3) * np.pi),
            "Radii (μm)": tuple([round(float(r), 2) for r in radii]),
        }
    else:
        result = None
    return result


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
def generate_batches(iterable, batch_size):
    """
    Yield successive batches from iterable.
    """
    for i in range(0, len(iterable), batch_size):
        n = min(i + batch_size, len(iterable) - 1)
        yield iterable[i:n]
