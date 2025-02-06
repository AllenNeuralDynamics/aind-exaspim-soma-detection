"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals from a whole-brain with a convolutional
neural network.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import euclidean
from skimage import exposure
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
        Unique identifier for the whole brain dataset.
    proposals : List[Tuple[float]]
        List of proposals, where each is represented by an xyz coordinate.
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
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


def run_inference(dataloader, model, device, verbose=True):
    """
    Runs inference on a given dataset using the provided model, then returns
    the predictions and ground truth labels.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader object that loads the dataset in batches.
    model : torch.nn.Module
        Neural network model that is used to generate predictions.
    device : str
        Name of device where model should be loaded and run.
    verbose : bool, optional, default=True
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
    keys, hat_y, y = list(), list(), list()
    with torch.no_grad():
        model.eval()
        n = dataloader.n_rounds
        iter = tqdm(dataloader, total=n, dynamic_ncols=True) if verbose else dataloader
        for keys_i, x_i, y_i in iter:
            # Forward pass
            x_i = x_i.to(device)
            hat_y_i = torch.sigmoid(model(x_i))

            # Store result
            keys.extend(keys_i)
            hat_y.append(ml_util.toCPU(hat_y_i))
            y.append(np.array(y_i) if y_i[0] is not None else list())
    return keys, np.vstack(hat_y)[:, 0], np.vstack(y)[:, 0]


def load_model(path, patch_shape, device):
    """
    Loads a pre-trained model from the given, then transfers the model to the
    specified device (i.e. CPU or GPU).

    Parameters
    ----------
    path : str
        Path to the saved model weights.
    patch_shape : Tuple[int]
        Shape of the input patches expected by the model expects.
    device : str
        Name of device where model should be loaded and run.

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
def branchiness_filtering(
    img_prefix, accepted_proposals, multiscale, patch_shape
):
    """
    Filters a list of accepted proposals by checking whether there exists a
    branch emanating from the center, see "branch_filtering" for details.

    Parameters
    ----------
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    accepted_proposals : List[Tuple[float]]
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
    # Initializations
    img = img_util.open_img(img_prefix)
    voxels = [img_util.to_voxels(p, multiscale) for p in accepted_proposals]
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for voxel in voxels:
            threads.append(
                executor.submit(is_branchy, img, voxel, patch_shape)
            )

        # Process results
        filtered_accepts = list()
        with tqdm(total=len(threads)) as pbar:
            for thread in as_completed(threads):
                branchy_bool, voxel = thread.result()
                if branchy_bool:
                    xyz = img_util.to_physical(voxel, multiscale)
                    filtered_accepts.append(xyz)
                pbar.update(1)
    return filtered_accepts


def is_branchy(img, voxel, patch_shape):
    # Fit Gaussian
    img_patch = np.minimum(img_util.get_patch(img, voxel, patch_shape), 400)
    _, params = spg.gaussian_fitness(img_patch, r=3)
    mean = tuple(params[0:3].astype(int))

    # Branchiness Check
    branch_dist = max(2.5 * np.sqrt(3 * np.min(params[3:6]**2)), 8)
    if branch_dist < patch_shape[0] // 2 - 1 and branch_dist > 0:
        img_patch = exposure.equalize_adapthist(img_patch, nbins=6)
        if branch_search(img_patch, mean, branch_dist):
            return True, voxel
        else:
            return False, voxel
    return True, voxel


def branch_search(img_patch, root, min_dist):
    if img_util.is_inbounds(root, img_patch.shape):
        # Run search
        max_dist = 0
        queue = [root]
        visited = set()
        while len(queue) > 0:
            # Visit voxel
            voxel = queue.pop()
            if euclidean(voxel, root) >= min_dist:
                return True
            if euclidean(voxel, root) > max_dist:
                max_dist = euclidean(voxel, root)
            visited.add(voxel)

            # Update queue
            for nb in img_util.get_nbs(voxel, img_patch.shape):
                if nb not in visited and img_patch[nb] > 0.15:
                    queue.append(nb)
        return False
    else:
        return True
