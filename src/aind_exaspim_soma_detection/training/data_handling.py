"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data and applying data augmentation (if applicable).

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import rotate
from torch.utils.data import Dataset

import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms

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

    def __init__(self, patch_shape, transform=False):
        # Initialize class attributes
        self.examples = dict()  # key: (brain_id, voxel), value: label
        self.imgs = dict()  # key: brain_id, value: image
        self.img_paths = dict()
        self.patch_shape = patch_shape

        # Data augmentation
        if transform:
            self.transform = transforms.Compose(
                [
                    RandomFlip3D(),
                    RandomNoise3D(),
                    RandomRotation3D(angles=(-30, 30)),
                    RandomContrast3D(factor_range=(0.7, 1.3)),
                    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(
                        0
                    ),
                ]
            )
        else:
            self.transform = transform

    def n_examples(self):
        """
        Counts the number of examples in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of examples in the dataset.

        """
        return len(self.examples)

    def n_positives(self):
        """
        Counts the number of positive examples in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of positive examples in the dataset.

        """
        return len(self.get_positives())

    def n_negatives(self):
        """
        Counts the number of negative examples in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of negative examples in the dataset.

        """
        return len(self.get_negatives())

    def get_positives(self):
        """
        Gets all positive examples in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Positive examples in dataset.

        """
        return dict({k: v for k, v in self.examples.items() if v})

    def get_negatives(self):
        """
        Gets all negative examples in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Negetaive examples in dataset.

        """
        return dict({k: v for k, v in self.examples.items() if not v})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        brain_id, voxel = key
        img_patch = img_util.get_patch(
            self.imgs[brain_id], voxel, self.patch_shape
        )
        return img_patch / 2**15, self.examples[key]

    def ingest_examples(self, brain_id, img_prefix, proposals, labels=None):
        # Load image
        if brain_id not in self.imgs:
            self.imgs[brain_id] = img_util.open_img(img_prefix)
            self.img_paths[brain_id] = img_prefix

        # Check if labels are valid
        if labels is not None:
            assert len(proposals) == len(labels), "#proposals != #labels"

        # Load proposals
        for i, voxel in enumerate(proposals):
            key = (brain_id, tuple(voxel))
            self.examples[key] = labels[i] if labels else None

    def visualize_example(self, key):
        img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

    def visualize_augmented_example(self, key):
        # Get image patch
        img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

        # Apply transforms
        img_patch = np.array(self.transform(img_patch))
        img_util.plot_mips(img_patch[0, ...], clip_bool=True)


# --- Data Augmentation ---
class RandomFlip3D:
    """
    Randomly flip a 3D image along one or more axes.

    """

    def __init__(self, axes=(0, 1, 2)):
        self.axes = axes

    def __call__(self, img):
        for axis in self.axes:
            if random.random() > 0.5:
                img = np.flip(img, axis=axis)
        return img


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.

    """

    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = np.random.normal(self.mean, self.std, img.shape)
        return img + noise


class RandomRotation3D:
    """
    Applies random rotation to a 3D image along a randomly chosen axis.

    """

    def __init__(self, angles=(-20, 20), axes=((0, 1), (0, 2), (1, 2))):
        self.angles = angles
        self.axes = axes

    def __call__(self, img):
        for _ in range(2):
            angle = random.uniform(*self.angles)
            axis = random.choice(self.axes)
            img = rotate(img, angle, axes=axis, reshape=False, order=1)
        return img


class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.

    """

    def __init__(self, factor_range=(0.7, 1.3)):
        self.factor_range = factor_range

    def __call__(self, img):
        factor = random.uniform(*self.factor_range)
        return np.clip(img * factor, img.min(), img.max())


# --- Custom Dataloader ---
class MultiThreadedDataLoader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size):
        """
        Constructs a multithreaded data loading object.

        Parameters
        ----------
        dataset : Dataset.SomaDataset
            Instance of custom dataset.
        batch_size : int
            Number of samples per batch.

        Returns
        -------
        None

        """
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self.DataLoaderIterator(self)

    class DataLoaderIterator:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
            self.batch_size = dataloader.batch_size
            self.keys = list(self.dataset.examples.keys())
            np.random.shuffle(self.keys)
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self):
            # Check whether to stop
            if self.current_index >= len(self.keys):
                raise StopIteration

            # Get the next batch of keys
            batch_keys = self.keys[
                self.current_index: self.current_index
                + self.dataloader.batch_size
            ]
            self.current_index += self.dataloader.batch_size

            # Load image patches
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = {
                    executor.submit(self.dataset.__getitem__, idx): idx
                    for idx in batch_keys
                }

                # Process results
                patches = list()
                labels = list()
                for thread in as_completed(threads):
                    patch, label = thread.result()
                    patches.append(torch.tensor(patch, dtype=torch.float))
                    labels.append(torch.tensor(label, dtype=torch.float))

            # Reformat inputs
            patches = torch.unsqueeze(torch.stack(patches), dim=1)
            labels = torch.unsqueeze(torch.stack(labels), dim=1)
            return patches, labels


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
