"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data and applying data augmentation (if applicable).

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import rotate
from torch.utils.data import Dataset

import ast
import numpy as np
import os
import random
import torch
import torchvision.transforms as transforms

from aind_exaspim_soma_detection.utils import img_util, util


# --- Custom Dataset ---
class ProposalDataset(Dataset):
    """
    A custom dataset used to train a neural network to classify soma proposals
    as either accept or reject. The dataset is initialized by providing
    the following inputs:
        (1) Path to a whole brain dataset stored in an S3 bucket
        (2) List of voxel coordinates representing soma proposals
        (3) Optionally, labels for each proposal (i.e. 0 or 1)

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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        brain_id, voxel = key
        img_patch = img_util.get_patch(
            self.imgs[brain_id], voxel, self.patch_shape
        )
        return key, img_patch / 2**15, self.examples[key]

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

    def ingest_examples(self, brain_id, img_prefix, proposals, labels=None):
        # Sanity check
        if labels is not None:
            assert len(proposals) == len(labels), "#proposals != #labels"

        # Load image (if applicable)
        if brain_id not in self.imgs:
            self.imgs[brain_id] = img_util.open_img(img_prefix)
            self.img_paths[brain_id] = img_prefix

        # Load proposal voxel coordinates
        for i, voxel in enumerate(proposals):
            key = (brain_id, tuple(voxel))
            self.examples[key] = labels[i] if labels else None

    def remove_example(self, key):
        del self.examples[key]

    def visualize_example(self, key):
        _, img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

    def visualize_augmented_example(self, key):
        # Get image patch
        _, img_patch, _ = self.__getitem__(key)
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

    def __init__(self, angles=(-45, 45), axes=((0, 1), (0, 2), (1, 2))):
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

    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, img):
        factor = random.uniform(*self.factor_range)
        return np.clip(img * factor, img.min(), img.max())


class RandomScale3D:
    """
    Randomly scale a 3D image along all axes.

    """

    def __init__(self, scale_range=(0.8, 1.2), axes=(0, 1, 2)):
        self.scale_range = scale_range
        self.axes = axes

    def __call__(self, img):
        # Generate scaling factors
        scales = list()
        for _ in self.axes:
            scales.append(
                random.uniform(self.scale_range[0], self.scale_range[1])
            )

        # Create a new shape by scaling the dimensions of the image
        new_shape = list(img.shape)
        for i, axis in enumerate(self.axes):
            new_shape[axis] = int(img.shape[axis] * scales[i])
        return np.resize(img, new_shape)


# --- Custom Dataloader ---
class MultiThreadedDataLoader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size, return_keys=False):
        """
        Constructs a multithreaded data loading object.

        Parameters
        ----------
        dataset : Dataset.ProposalDataset
            Instance of custom dataset.
        batch_size : int
            Number of samples per batch.
        return_keys : bool, optional
            Indication of whether to return example ids. The default is False.

        Returns
        -------
        None

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_keys = return_keys

    def __iter__(self):
        return self.DataLoaderIterator(self)

    class DataLoaderIterator:
        def __init__(self, dataloader):            
            self.current_index = 0
            self.batch_size = dataloader.batch_size
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
            self.keys = list(self.dataset.examples.keys())
            self.return_keys = dataloader.return_keys
            np.random.shuffle(self.keys)

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
                keys = list()
                patches = list()
                labels = list()
                for thread in as_completed(threads):
                    key, patch, label = thread.result()
                    keys.append(key)
                    patches.append(torch.tensor(patch, dtype=torch.float))
                    labels.append(torch.tensor(label, dtype=torch.float))

            # Reformat inputs
            patches = torch.unsqueeze(torch.stack(patches), dim=1)
            labels = torch.unsqueeze(torch.stack(labels), dim=1)
            if self.return_keys:
                return keys, patches, labels
            else:
                return patches, labels


# --- utils ---
def init_subdataset(dataset, positives, negatives, patch_shape, transform):
    subdataset = ProposalDataset(patch_shape, transform=transform)
    for example_tuple in merge_examples(dataset, positives, negatives):
        subdataset.ingest_examples(*example_tuple)
    return subdataset


def merge_examples(soma_dataset, positives, negatives):
    examples = list()
    combined_dict = positives.copy()
    combined_dict.update(negatives)
    for key, value in combined_dict.items():
        brain_id, voxel = key
        img_path = soma_dataset.img_paths[brain_id]
        examples.append((brain_id, img_path, [voxel], [value]))
    return examples
