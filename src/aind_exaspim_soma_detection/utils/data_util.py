"""
Created on Mon Jan 6 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading training data.

"""

import ast
import os

from aind_exaspim_soma_detection.utils import img_util, util


# --- Fetch Data ---
def load_examples(path):
    test_examples = list()
    for line in util.read_txt(path):
        idx = line.find(',')
        brain_id = ast.literal_eval(line[1:idx])
        xyz = ast.literal_eval(line[idx+2:-1])
        test_examples.append((brain_id, xyz))
    return test_examples


def fetch_smartsheet_somas(
    smartsheet_path, img_prefixes_path, multiscale, adjust_coords_bool=True
):
    # Read data
    soma_coords = util.extract_somas_from_smartsheet(smartsheet_path)
    img_prefixes = util.read_json(img_prefixes_path)

    # Reformat data
    data = list()
    for brain_id, xyz_list in soma_coords.items():
        if brain_id not in ["686955", "708373"]:
            data.append(
                reformat_data(brain_id, img_prefixes, multiscale, xyz_list, 1)
            )
    return data


def fetch_exaspim_somas_2024(dataset_path, img_prefixes_path, multiscale):
    """
    Fetches and formats soma data for training from the exaSPIM dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory containing brain-specific subdirectories
        with "accepts" and "rejects" folders.
    img_prefixes_path : str
        Path to a JSON file that maps brain IDs to image S3 prefixes.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinates must index into.

    Returns
    -------
    List[tuple]
        List of tuples where each tuple contains the following:
            - "brain_id" (str): Unique identifier for the brain.
            - "img_path" (str): Path to image stored in S3 image.
            - "voxels" (list): Voxel coordinates of proposed somas.
            - "labels" (list): Labels corresponding to voxels.

    """
    data = list()
    img_prefixes = util.read_json(img_prefixes_path)
    for brain_id in util.list_subdirectory_names(dataset_path):
        # Read data
        accepts_dir = os.path.join(dataset_path, brain_id, "accepts")
        accepts_xyz = util.read_swc_dir(accepts_dir)

        rejects_dir = os.path.join(dataset_path, brain_id, "rejects")
        rejects_xyz = util.read_swc_dir(rejects_dir)

        # Reformat data
        data.append(
            reformat_data(brain_id, img_prefixes, multiscale, accepts_xyz, 1)
        )
        data.append(
            reformat_data(brain_id, img_prefixes, multiscale, rejects_xyz, 0)
        )
    return data


def reformat_data(brain_id, img_prefixes, multiscale, xyz_list, label):
    """
    Reformats data for training or inference by converting xyz to voxel
    coordinates and associates them with a brain id, image path, and labels.

    Parameters
    ----------
    brain_id : str
        Unique identifier for the whole brain dataset.
    img_prefixes : dict
        A dictionary mapping brain IDs to image S3 prefixes.
    multiscale : int
        Level in the image pyramid that the voxel coordinates must index into.
    xyz_list : List[ArrayLike]
        List 3D xyz coordinates.
    label : int
        Label associated with the given coordinates (i.e. 1 for "accepts" and
        0 for "rejects").

    Returns
    -------
    tuple
        Tuple containing the "brain_id", "image_path", "voxels", and "labels".

    """
    img_path = img_prefixes[brain_id] + str(multiscale)
    voxels = [img_util.to_voxels(xyz, multiscale) for xyz in xyz_list]
    labels = len(voxels) * [label]
    return (brain_id, img_path, voxels, labels)
