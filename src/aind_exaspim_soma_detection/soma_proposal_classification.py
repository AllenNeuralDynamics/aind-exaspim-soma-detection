"""
Created on Wed Jan 8 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that classifies soma proposals.

"""

from tqdm import tqdm

import torch

from aind_exaspim_soma_detection.utils import img_util
from aind_exaspim_soma_detection.machine_learning.models import Fast3dCNN
from aind_exaspim_soma_detection.machine_learning.data_handling import MultiThreadedDataLoader, ProposalDataset


def classify_proposals(
    brain_id,
    proposals,
    img_prefix,
    patch_shape,
    multiscale,
    confidence_threshold,
    model_path,
    batch_size=16,
    device="cuda",
):
    # Initialize dataset
    proposals = [img_util.to_voxels(p, multiscale) for p in proposals]
    dataset = ProposalDataset(patch_shape)
    dataset.ingest_proposals(brain_id, img_prefix, proposals)

    # Main
    dataloader = MultiThreadedDataLoader(dataset, batch_size, True)
    model = load_model(model_path, patch_shape, device)
    somas = run_inference(dataloader, model, confidence_threshold, device)

    # Convert soma coordinates

    return somas

def run_inference(dataloader, model, confidence_threshold, device):
    somas = list()
    with torch.no_grad():
        for keys_i, x_i, _ in tqdm(dataloader):
            # Forward pass
            x_i = x_i.to(device)
            hat_y_i = torch.sigmoid(model(x_i))

            # Get positives


def load_model(path, patch_shape, device):
    model = Fast3dCNN(patch_shape)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model
