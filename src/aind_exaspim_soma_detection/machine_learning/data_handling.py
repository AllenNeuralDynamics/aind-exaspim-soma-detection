"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data and applying data augmentation (if applicable).

"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate, zoom
from scipy.spatial import distance
from torch.utils.data import Dataset

from aind_exaspim_soma_detection.utils import img_util


# --- Custom Dataset ---
class ProposalDataset(Dataset):
    """
    Custom dataset for classifying soma proposals as either accept or reject
    by using a neural network. Proposals are stored in the "self.proposals"
    dictionary, where each item consists of the following:
        - Key: (brain_id, voxel)
        - Value: label of proposal (0 or 1), optional

    This dataset is populated using the "self.ingest_proposals" method, which
    requires the following inputs:
        (1) brain_id: Unique identifier of brain containing proposals.
        (2) img_prefix: Path to whole-brain image stored in an S3 bucket.
        (3) voxels: List of voxel coordinates of proposals.
        (4) labels: Labels for each proposal (0 or 1), optional.

    Note: This dataset supports proposals from multiple whole-brain datasets.

    """

    def __init__(self, patch_shape, transform=False):
        """
        Initializes a custom dataset for processing soma proposals.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of the image patches to be extracted centered at proposals.
        transform : bool, optional
            Indication of whether to apply data augmentation to image patches.
            The default is False.

        Attributes
        ----------
        proposals : dict
            Dictionary where each key is a tuple "(brain_id, voxel)" and the
            value is the corresponding label of the proposal (0 or 1).
        imgs : dict
            Dictionary where each key is a "brain_id" and the value is the
            corresponding whole-brain image.
        img_paths : dict
            A dictionary for storing the paths to the whole-brain images.
        patch_shape : Tuple[int]
            Shape of the image patches to be extracted centered at proposals.
        transform : callable or None
            Transformation pipeline applied to each image patch if "transform"
            is True. Otherwise, this value is set to None.

        """
        # Class attributes
        self.key_to_filename = dict()
        self.proposals = dict()
        self.imgs = dict()
        self.img_paths = dict()
        self.patch_shape = patch_shape

        # Data augmentation (if applicable)
        if transform:
            self.transform = transforms.Compose(
                [
                    RandomFlip3D(),
                    RandomRotation3D(),
                    RandomScale3D(),
                    RandomContrast3D(),
                    RandomBrightness3D(),
                    RandomNoise3D(),
                    lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(
                        0
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        """
        Counts the number of proposals in self.

        Parameters
        ----------
        None

        Returns
        -------
        Number of proposals in self.

        """
        return len(self.proposals)

    def __getitem__(self, key):
        """
        Gets the proposal corresponding to the given key, which consists of a
        "brain_id" and "voxel" coordinate. The proposal includes a normalized
        image patch centered at the voxel, along with its corresponding label.

        Parameters
        ----------
        key : tuple
            A tuple containing:
            - "brain_id" (str): Unique identifier of the brain dataset.
            - "voxel" (Tuple[int]): Voxel coordinate of proposal.

        Returns:
        --------
        Tuple
            A tuple containing:
            - "key" (tuple): Input "key" tuple.
            - "img_patch" (numpy.ndarray): 3D image patch centered at "voxel".
            - "label" (int): Label associated with the proposal.

        """
        # Get voxel
        brain_id, voxel = key
        if self.transform:
            voxel = [voxel_i + random.randint(-5, 5) for voxel_i in voxel]

        # Get image patch
        try:
            img_patch = img_util.get_patch(
                self.imgs[brain_id], voxel, self.patch_shape
            )
            img_patch = img_util.normalize(img_patch)
        except:
            print(voxel, self.imgs[brain_id].shape)
            img_patch = np.ones(self.patch_shape)
        return key, img_patch, self.proposals[key]

    def get_positives(self):
        """
        Gets all positive proposals in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Positive proposals in dataset.

        """
        return dict({k: v for k, v in self.proposals.items() if v})

    def get_negatives(self):
        """
        Gets all negative proposals in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Negetaive proposals in dataset.

        """
        return dict({k: v for k, v in self.proposals.items() if not v})

    def n_positives(self):
        """
        Counts the number of positive proposals in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of positive proposals in the dataset.

        """
        return len(self.get_positives())

    def n_negatives(self):
        """
        Counts the number of negative proposals in the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of negative proposals in the dataset.

        """
        return len(self.get_negatives())

    def ingest_proposals(
        self, brain_id, img_prefix, voxels, labels=None, filenames=None
    ):
        # Sanity checks
        if labels is not None:
            assert len(voxels) == len(labels), "#proposals != #labels"
        if filenames is not None:
            assert len(voxels) == len(filenames), "#proposals != #filenames"

        # Load image (if applicable)
        if brain_id not in self.imgs:
            self.imgs[brain_id] = img_util.open_img(img_prefix)
            self.img_paths[brain_id] = img_prefix

        # Load proposal voxel coordinates
        for i, voxel in enumerate(voxels):
            key = (brain_id, tuple(voxel))
            self.proposals[key] = labels[i] if labels else -1
            if filenames is not None:
                self.key_to_filename[key] = filenames[i]

    def remove_proposal(self, query_key, epsilon=0):
        # Remove if proposal exists
        if query_key in self.proposals:
            del self.proposals[query_key]

        # Search for nearby proposal
        if epsilon > 0:
            query_brain_id, query_voxel = query_key
            for brain_id, voxel in self.proposals:
                d = distance.euclidean(query_voxel, voxel)
                if brain_id == query_brain_id and d < epsilon:
                    del self.proposals[(brain_id, voxel)]
                    break

    def visualize_proposal(self, key):
        _, img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

    def visualize_augmented_proposal(self, key):
        # Get image patch
        _, img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

        # Apply transforms
        img_patch = np.array(self.transform(img_patch))
        img_util.plot_mips(img_patch[0, ...], clip_bool=True)


# --- Data Augmentation ---
class RandomBrightness3D:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, img):
        factor = 1 + np.random.uniform(-self.delta, self.delta)
        return img * factor


class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.

    """

    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, img):
        factor = random.uniform(*self.factor_range)
        return np.clip(img * factor, img.min(), img.max())


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

    def __init__(self, mean=0.0, std=0.025):
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

    def __call__(self, img, mode="grid-mirror"):
        for axis in self.axes:
            angle = random.uniform(*self.angles)
            img = rotate(
                img, angle, axes=axis, mode=mode, reshape=False, order=1
            )
        return img


class RandomScale3D:
    """
    Applies random scaling to an image along each axis.

    """

    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, img):
        # Sample new image shape
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_shape = (
            int(img.shape[0] * alpha),
            int(img.shape[1] * alpha),
            int(img.shape[2] * alpha),
        )

        # Compute the zoom factors
        shape = img.shape
        zoom_factors = [
            new_dim / old_dim for old_dim, new_dim in zip(shape, new_shape)
        ]
        return zoom(img, zoom_factors, order=3)


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
        dataset : Dataset.ProposalDataset
            Instance of custom dataset.
        batch_size : int
            Number of samples per batch.

        Returns
        -------
        None

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_rounds = len(dataset) // batch_size

    def __iter__(self):
        return self.DataLoaderIterator(self)

    class DataLoaderIterator:
        def __init__(self, dataloader):
            self.current_index = 0
            self.batch_size = dataloader.batch_size
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
            self.keys = list(self.dataset.proposals.keys())
            np.random.shuffle(self.keys)

        def __iter__(self):
            return self

        def __next__(self):
            # Check whether to stop
            if self.current_index >= len(self.keys):
                raise StopIteration

            # Get the next batch of proposals
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
            return keys, patches, labels


# --- utils ---
def init_subdataset(dataset, positives, negatives, patch_shape, transform):
    subdataset = ProposalDataset(patch_shape, transform=transform)
    for proposal_tuple in merge_proposals(dataset, positives, negatives):
        subdataset.ingest_proposals(*proposal_tuple)
    return subdataset


def merge_proposals(soma_dataset, positives, negatives):
    proposals = list()
    combined_dict = positives.copy()
    combined_dict.update(negatives)
    for key, value in combined_dict.items():
        brain_id, voxel = key
        img_path = soma_dataset.img_paths[brain_id]
        proposals.append((brain_id, img_path, [voxel], [value]))
    return proposals
