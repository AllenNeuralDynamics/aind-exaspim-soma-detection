"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals with a convolutional neural network.

"""

from tqdm import tqdm

import numpy as np
import torch

from aind_exaspim_soma_detection.utils import img_util, ml_util
from aind_exaspim_soma_detection.machine_learning.models import Fast3dCNN
from aind_exaspim_soma_detection.machine_learning.data_handling import MultiThreadedDataLoader, ProposalDataset


def classify_proposals(
    brain_id,
    proposals,
    img_prefix,
    patch_shape,
    multiscale,
    model_path,
    threshold,
    batch_size=16,
    device="cuda",
):
    # Initialize dataset
    proposals = [img_util.to_voxels(p, multiscale) for p in proposals]
    dataset = ProposalDataset(patch_shape)
    dataset.ingest_proposals(brain_id, img_prefix, proposals)

    # Generate predictions
    dataloader = MultiThreadedDataLoader(dataset, batch_size, True)
    model = load_model(model_path, patch_shape, device)
    keys, hat_y, _ = run_inference(dataloader, model, device)

    # Extract predicted somas
    soma_xyz_list = list()
    for key_i, hat_y_i in zip(keys, hat_y):
        if hat_y_i > threshold:
            voxel = key_i[1] * 2**multiscale
            soma_xyz_list.append(img_util.to_physical(voxel[::-1]))
    return soma_xyz_list


def run_inference(dataloader, model, device, progress_bool=True):
    model.eval()
    keys, hat_y, y = list(), list(), list()
    with torch.no_grad():
        iterator = tqdm(dataloader) if progress_bool else dataloader
        for keys_i, x_i, y_i in iterator:
            # Forward pass
            x_i = x_i.to(device)
            hat_y_i = torch.sigmoid(model(x_i))

            # Store result
            keys.extend(keys_i)
            hat_y.append(ml_util.toCPU(hat_y_i))
            y.append(np.array(y_i) if y_i[0] is not None else list())
    return keys, np.vstack(hat_y), np.vstack(y)


def load_model(path, patch_shape, device):
    model = Fast3dCNN(patch_shape)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model
