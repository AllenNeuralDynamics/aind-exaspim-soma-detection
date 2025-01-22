"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals from a whole-brain with a convolutional
neural network.

"""

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
        Device to run the inference on. The default is "cuda".

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
        A DataLoader object that loads the dataset in batches.
    model : torch.nn.Module
        Neural network model that is used to generate predictions.
    device : str
        Device to run the inference on. The default is "cuda".
    verbose : bool, optional, default=True
        Indication of whether to display a progress bar during inference.

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
        iterator = tqdm(dataloader, total=n) if verbose else dataloader
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
