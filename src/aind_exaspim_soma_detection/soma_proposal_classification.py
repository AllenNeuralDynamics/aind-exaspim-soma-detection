"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals from a whole-brain with a convolutional
neural network.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist, euclidean
from sklearn.cluster import KMeans
from tqdm import tqdm

import numpy as np
import torch

from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import img_util, ml_util
from aind_exaspim_soma_detection.machine_learning.models import FastConvNet3d
from aind_exaspim_soma_detection.machine_learning.data_handling import (
    MultiThreadedDataLoader,
    ProposalDataset,
)


def classify_proposals(
    brain_id,
    proposals,
    img_prefix,
    model_path,
    multiscale,
    patch_shape,
    threshold,
    batch_size=16,
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
    img_prefix : str
        Prefix (or path) of a whole-brain image stored in a S3 bucket.
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
        List of physical coordinates of the somas detected by the model.

    """
    # Initialize dataset
    proposals = [img_util.to_voxels(p, multiscale) for p in proposals]
    dataset = ProposalDataset(patch_shape)
    dataset.ingest_proposals(brain_id, img_prefix, proposals)

    # Generate predictions
    dataloader = MultiThreadedDataLoader(dataset, batch_size)
    model = load_model(model_path, patch_shape, device)
    keys, hat_y, _ = run_inference(dataloader, model, device)

    # Extract predicted somas
    soma_xyz_list = list()
    for key_i, hat_y_i in zip(keys, hat_y):
        if hat_y_i > threshold:
            soma_xyz_list.append(img_util.to_physical(key_i[1], multiscale))
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
            - "keys" (List[tuple]): Unique identifier for each proposal that
               consists of the "brain_id" and "voxel" coordinate.
            - "hat_y" (numpy.ndarray): Prediction for each proposal.
            - "y" (numpy.ndarray): Ground truth label for each proposal.

    """
    # Initializations
    n = dataloader.n_rounds
    iterator = tqdm(dataloader, total=n) if verbose else dataloader

    # Main
    keys, hat_y, y = list(), list(), list()
    with torch.no_grad():
        model.eval()
        for keys_i, x_i, y_i in iterator:
            # Forward pass
            x_i = x_i.to(device)
            hat_y_i = torch.sigmoid(model(x_i))

            # Store result
            keys.extend(keys_i)
            hat_y.append(ml_util.toCPU(hat_y_i))
            y.append(np.array(y_i) if y_i[0] is not None else list())
    return keys, np.vstack(hat_y)[:, 0], np.vstack(y)[:, 0]


def load_model(path, patch_shape, device="cuda"):
    """
    Loads a pre-trained model from the given, then transfers the model to the
    specified device (i.e. CPU or GPU).

    Parameters
    ----------
    path : str
        Path to the saved model weights.
    patch_shape : Tuple[int]
        Shape of the input patches expected by the model expects.
    device : str, optional
        Name of device where model should be loaded and run. The default is
        "cuda".

    Returns
    -------
    FastConvNet3d
        Model instance with the loaded weights.

    """
    model = FastConvNet3d(patch_shape)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    return model


# --- Accepted Proposal Filtering ---
def compute_scores(score_func, img, voxels, patch_shape):
    voxel_list, score_list = list(), list()
    for voxel in tqdm(voxels):
        voxel, score = score_func(img, voxel, patch_shape)
        voxel_list.append(voxel)
        score_list.append(score)
    return voxel_list, score_list

    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for voxel in voxels:
            threads.append(
                executor.submit(score_func, img, voxel, patch_shape)
            )

        # Process results
        voxels, scores = list(), list()
        pbar = tqdm(total=len(threads)) 
        for thread in as_completed(threads):
            voxel, score = thread.result()
            if score is not None:
                voxels.append(voxel)
                scores.append(score)
            pbar.update(1)
    return voxels, scores


def branchiness_filtering(
    img_prefix,
    accepts,
    multiscale,
    patch_shape,
    min_branchiness_score=25
):
    """
    Filters a list of accepted proposals by checking whether there exists a
    branch emanating from the center, see "branch_filtering" for details.

    Parameters
    ----------
    img_prefix : str
        Prefix (or path) of a whole-brain image stored in a S3 bucket.
    accepts : List[Tuple[float]]
        List of accepted proposals, where each is represented by an xyz
        coordinate.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.
    patch_shape : Tuple[int]
        Shape of the image patches to be read in order to check for branches.

    Returns
    -------
    List[Tuple[int]]
        List of accepted proposals that have a branch emanating from the
        center.

    """
    # Compute scores
    img = img_util.open_img(img_prefix)
    voxels = [img_util.to_voxels(p, multiscale) for p in accepts]
    voxels, scores = compute_scores(
        compute_branchiness, img, voxels, patch_shape
    )

    # Process results
    filtered_accepts, filtered_scores = list(), list()
    for voxel, score in zip(voxels, scores):
        branchiness, brightness = score
        if branchiness > min_branchiness_score:
            filtered_accepts.append(img_util.to_physical(voxel, multiscale))
            filtered_scores.append(score)
    return filtered_accepts


def compute_branchiness(img, voxel, patch_shape, return_mask=False):
    center = tuple([s // 2 for s in patch_shape])
    img_patch = np.minimum(img_util.get_patch(img, voxel, patch_shape), 1000)
    branchiness, brightness, object_mask = branch_search(img_patch, center)
    if return_mask:
        return voxel, branchiness, brightness, object_mask
    else:
        return voxel, (branchiness, brightness)


def branch_search(img_patch, root):
    # Compute foreground
    relabeled = kmeans_intensity_clustering(img_patch)
    foreground = (relabeled > 0).astype(int)

    # Search center object
    max_dist = 0
    object_mask = np.zeros_like(foreground)
    queue = [root]
    visited = set([root])
    while len(queue) > 0:
        # Visit voxel
        voxel = queue.pop()
        max_dist = max(euclidean(voxel, root), max_dist)
        object_mask[voxel] = 1

        # Update queue
        for nb in img_util.get_nbs(voxel, img_patch.shape):
            if nb not in visited and foreground[nb] > 0:
                queue.append(nb)
                visited.add(nb)

    # Process results
    brightness = np.sum(object_mask * img_patch)
    return max_dist, brightness, object_mask


def brightness_filtering(
    img_prefix, accepts, multiscale, patch_shape, max_accepts=800
):
    # Compute scores
    img = img_util.open_img(img_prefix)
    voxels = [img_util.to_voxels(p, multiscale) for p in accepts]
    voxels, scores = compute_scores(
        compute_brightness, img, voxels, patch_shape
    )

    # Process results
    idxs = np.flip(np.argsort(scores))[0:max_accepts]
    return np.array(accepts)[idxs]


def compute_brightness(img, voxel, patch_shape):
    # Read image patch
    img_patch = img_util.get_patch(img, voxel, patch_shape)
    _, params = spg.gaussian_fitness(img_patch)
    voxels = reformat_coords(spg.generate_grid_coords(img_patch.shape))
    mean = np.array(params[0:3]).reshape(1, -1)

    # Compute score
    distances = cdist(voxels, mean, metric='euclidean')
    std_dist = np.sqrt(2 * np.sum(np.min(params[3:6])**2))
    within_one_sigma = distances < std_dist
    img_vals = img_patch.flatten()[within_one_sigma.flatten()]
    score = np.percentile(img_vals, 80) if len(img_vals) > 0 else np.inf
    return voxel, score


# --- Helpers ---
def kmeans_intensity_clustering(img_patch, n_clusters=3):
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=15)
        kmeans.fit(img_patch.reshape(-1, 1))
        return kmeans.labels_.reshape(img_patch.shape)
    except:
        return np.ones_like(img_patch)


def reformat_coords(coords):
    coord_list = list()
    for i in range(len(coords[0])):
        coord_list.append((coords[0][i], coords[1][i], coords[2][i]))
    return np.array(coord_list)
