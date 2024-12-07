"""
Created on Mon Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


"""

from torch.utils.data import Dataset

import os

from aind_exaspim_soma_detection.utils import img_util, util


# --- Custom Dataset ---
class SomaDataset(Dataset):
    """
    A custom dataset used to train a neural network to classify soma proposals
    as either accepted or rejected. The dataset is initialized by providing
    the following inputs:
        (1) Path to a whole brain dataset stored in an S3 bucket
        (2) List of voxel coordinates representing soma proposals
        (3) Optionally, labels for each proposal (i.e. accept or reject)

    Note: This dataset supports inputs from multiple whole brain datasets.

    """

    def __init__(self, patch_shape):
        # Initialize class attributes
        self.examples = dict()  # keys: (brain_id, voxel), values: label
        self.imgs = dict()  # keys: brain_id, values: image
        self.patch_shape = patch_shape

    def __len__(self):
        """
        Gets the number of examples in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of examples in dataset.

        """
        return len(self.examples)

    def n_positive_examples(self):
        pass

    def n_negative_examples(self):
        pass

    def __getitem__(self, key):
        brain_id, voxel = key
        return img_util.get_patch(self.imgs[brain_id], voxel, self.patch_shape)

    def ingest_examples(self, brain_id, img_prefix, proposals, labels=None):
        # Load image
        self.imgs[brain_id] = img_util.open_img(img_prefix)

        # Check if labels are valid
        if labels is not None:
            assert len(proposals) != len(labels), "#proposals != #labels"

        # Load proposals
        for i, voxel in enumerate(proposals):
            key = (brain_id, tuple(voxel))
            self.examples[key] = labels[i] if labels else None


# --- Custom Data Loader ---
class SomaDataLoader:
    def __init__(self, apply_augmentation=False):
        # Initialize class attributes
        self.apply_augmentation = apply_augmentation


# --- Fetch Training Data ---
def fetch_smartsheet_somas(smartsheet_path, img_prefixes_path, multiscale=0):
    # Read data
    soma_coords = util.extract_somas_from_smartsheet(smartsheet_path)
    img_prefixes = util.read_json(img_prefixes_path)

    # Reformat data
    data = list()
    for brain_id, xyz_list in soma_coords.items():
        data.append(
            reformat_data(brain_id, img_prefixes, multiscale, xyz_list, 1)
        )
    return data


def fetch_exaspim_somas_2024(dataset_path, img_prefixes_path, multiscale=0):
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
            - "brain_id" (str): The unique identifier for the brain.
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
        List 3D coordinates (x, y, z).
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
