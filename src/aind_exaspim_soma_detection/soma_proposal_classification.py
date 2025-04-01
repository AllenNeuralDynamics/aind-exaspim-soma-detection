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
        Prefix (or path) of a whole-brain image stored in a S3 bucket.
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
                voxel, branchy_bool = thread.result()
                if branchy_bool:
                    xyz = img_util.to_physical(voxel, multiscale)
                    filtered_accepts.append(xyz)
                pbar.update(1)
    return filtered_accepts


def is_branchy(img, voxel, patch_shape, branch_dist=10):
    center = tuple([s // 2 for s in patch_shape])
    try:
        img_patch = img_util.get_patch(img, voxel, patch_shape)
        img_patch = np.minimum(img_util.get_patch(img, voxel, patch_shape), 250)
    
        fg_brightness = branch_search(img_patch, center, branch_dist)
        bg_brightness = np.percentile(img_patch, 20)
        if fg_brightness:
            contrast_score_1 = np.mean(fg_brightness) - bg_brightness
            contrast_score_2 = np.mean(fg_brightness) / bg_brightness
            return voxel, contrast_score_1 >= 50 or contrast_score_2 > 5
        else:
            return voxel, False
    except:
        print(f"Failed on {center} for img.shape {img.shape}")
        return voxel, False


def branch_search(img_patch, root, min_dist):
    # Initializations
    binarized = exposure.equalize_adapthist(img_patch, nbins=6) > 0.15
    fg_brightness = list()
    max_dist = 0

    # Search
    if img_util.is_inbounds(root, img_patch.shape):
        queue = [root]
        visited = set({root})
        while len(queue) > 0:
            # Visit voxel
            voxel = queue.pop()
            max_dist = max(max_dist, euclidean(voxel, root))
            if euclidean(voxel, root) >= min_dist:
                fg_brightness.append(img_patch[voxel])
    
            # Update queue
            for nb in img_util.get_nbs(voxel, img_patch.shape):
                if nb not in visited and binarized[nb]:
                    queue.append(nb)
                    visited.add(nb)
    return fg_brightness


def is_branchy_old(img, voxel, patch_shape, branch_dist=20.0):
    """
    Checks whether the soma at the given voxel is "branchy", meaning there
    exists a branch with length "branch_dist" microns extending from the soma.

    Parameters
    ----------
    img : zarr.core.Array
        Array representing a 3D image of a whole-brain.
    voxel : Tuple[int]
        Coordinate that represents the location of a soma.
    patch_shape : Tuple[int]
        Shape of the image patch to be extracted from "img" which is centered
        at "voxel".
    branch_dist : float, optional
        Distance from center that determines if a detected somas is branchy.
        The default is 20.0.

    Returns
    -------
    tuple
        A tuple that contains the following:
            - voxel (Tuple[int]): Location of a soma.
            - is_branchy (bool) : Indication of whether soma is branchy.
    """
    center = tuple([s // 2 for s in patch_shape])
    try:
        img_patch = np.minimum(img_util.get_patch(img, voxel, patch_shape), 250)
        img_patch = exposure.equalize_adapthist(img_patch, nbins=6)
        return voxel, branch_search(img_patch, center, branch_dist)
    except:
        print(f"Failed on center {center} for img.shape {img.shape}")
        return voxel, False


def branch_search_old(img_patch, root, min_dist):
    """
    Performs a breadth-first search (BFS) on a 3D image patch to check if
    there is a voxel in the foreground object containing "root". Note: this
    routine is used to check if a soma (i.e. root) is branchy.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image patch to be searched.
    root : Tuple[int]
        Voxel coordinates which is the root of the BFS.
    min_dist : float
        Distance from root that determines if the corresponding foreground
        object is branchy.

    Returns
    -------
    bool
        Indication of whether the foreground object containing root is
        branchy.

    """
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
