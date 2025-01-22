"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.spatial import distance
from torch.utils.data import Dataset

import numpy as np
import random
import torch
import torchvision.transforms as transforms

from aind_exaspim_soma_detection.machine_learning.augmentation import (
    RandomContrast3D,
    RandomFlip3D,
    RandomNoise3D,
    RandomRotation3D,
    RandomScale3D,
)
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
        Gets the image patch centered at the voxel corresponding to the given
        key.

        Parameters
        ----------
        key : tuple
            Unique indentifier of a proposal which is a tuple containing:
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
            Negative proposals in dataset.

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
        self, brain_id, img_prefix, voxels, labels=None, paths=None
    ):
        """
        Ingests proposals represented by voxel coordinates along with optional
        labels and paths into the dataset.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the whole-brain dataset.
        img_prefix : str
            Prefix (or path) of a whole-brain image stored in a S3 bucket.
        voxels : List[Tuple[int]]
            List of voxel coordinates representing the proposals.
        labels : List[int], optional
            List of ground truth labels corresponding to the proposals. The
            default is None.
        paths : List[str], optional
            List of file paths corresponding to the proposal. The default is
            None.

        Returns
        -------
        None

        """
        # Sanity checks
        if labels is not None:
            assert len(voxels) == len(labels), "#proposals != #labels"
        if paths is not None:
            assert len(voxels) == len(paths), "#proposals != #paths"

        # Load image (if applicable)
        if brain_id not in self.imgs:
            self.imgs[brain_id] = img_util.open_img(img_prefix)
            self.img_paths[brain_id] = img_prefix

        # Load proposal voxel coordinates
        for i, voxel in enumerate(voxels):
            key = (brain_id, tuple(voxel))
            self.proposals[key] = labels[i] if labels else -1
            if paths is not None:
                self.key_to_filename[key] = paths[i]

    def remove_proposal(self, query_key, epsilon=0):
        """
        Removes the proposal corresponding to the given key. Optionally,
        removes nearby proposals within a specified distance "epsilon".

        Parameters
        ----------
        query_key : tuple
            Unique indentifier of a proposal which is a tuple containing:
            - "brain_id" (str): Unique identifier of the brain dataset.
            - "voxel" (Tuple[int]): Voxel coordinate of proposal.
        epsilon : float, optional
            Distance threshold used to search for nearby proposals.

        Returns
        -------
        None

        """
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
        """
        Plots maximum intensity projections (MIPs) of the image patch centered
        at the proposal corresponding to the given key.

        Parameters
        ----------
        key : tuple
            Unique indentifier of a proposal which is a tuple containing:
            - "brain_id" (str): Unique identifier of the brain dataset.
            - "voxel" (Tuple[int]): Voxel coordinate of proposal.

        Returns
        -------
        None

        """
        _, img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

    def visualize_augmented_proposal(self, key):
        """
        Plots maximum intensity projections (MIPs) of the image patch centered
        at the given proposal key, before and after image augmentation.

        Parameters
        ----------
        key : tuple
            Unique indentifier of a proposal which is a tuple containing:
            - "brain_id" (str): Unique identifier of the brain dataset.
            - "voxel" (Tuple[int]): Voxel coordinate of proposal.

        Returns
        -------
        None

        """
        # Get image patch
        _, img_patch, _ = self.__getitem__(key)
        img_util.plot_mips(img_patch, clip_bool=True)

        # Apply transforms
        img_patch = np.array(self.transform(img_patch))
        img_util.plot_mips(img_patch[0, ...], clip_bool=True)


# --- Custom Dataloader ---
class MultiThreadedDataLoader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Constructs a multithreaded data loader.

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
        """
        Returns an iterator for the data loader, providing the functionality
        to iterate over the whole dataset.

        Returns
        -------
        iterator
            Iterator for the data loader.

        """
        return self.DataLoaderIterator(self)

    class DataLoaderIterator:
        """
        Custom iterator class for iterating over the dataset in the data
        loader.

        """

        def __init__(self, dataloader):
            """
            Initializes the "DataLoaderIterator" object for custom iteration
            over the dataset.

            Parameters
            ----------
            MultiThreadedDataLoader
                Data loader instance that this iterator is associated with.

            """
            self.current_index = 0
            self.batch_size = dataloader.batch_size
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
            self.keys = list(self.dataset.proposals.keys())
            np.random.shuffle(self.keys)

        def __iter__(self):
            """
            Returns the iterator object for custom iteration over the dataset.

            Returns
            -------
            DataLoaderIterator
                Iterator object itself.

            """
            return self

        def __next__(self):
            """
            Retrieves the next batch of image patches and their corresponding
            labels from the dataset.

            Parameters
            ----------
            None

            Returns
            -------
            tuple
                A tuple containing the following:
                - "keys" (list): List of keys corresponding to the current
                  batch of proposals.
                - "patches" (torch.Tensor): Image patches from the dataset
                  with the shape (self.batch_size, 1, H, W, D).
                - "labels" (torch.Tensor): Labels corresponding to the image
                   patches with the shape (self.batch_size, 1).

            """
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
