"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset

import numpy as np
import os
import random
import torch

from aind_exaspim_soma_detection.machine_learning.augmentation import (
    ImageTransforms,
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
        (2) img_path: Path to whole-brain image stored in an S3 bucket.
        (3) voxels: List of voxel coordinates of proposals.
        (4) labels: Labels for each proposal (0 or 1), optional.

    Note: This dataset supports proposals from multiple whole-brain datasets.
    """

    def __init__(
        self,
        patch_shape,
        brightness_clip=300,
        multiscale=0,
        normalization_percentiles=(1, 99.5),
        transform=False,
    ):
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
        self.brightness_clip = brightness_clip
        self.imgs = dict()
        self.img_paths = dict()
        self.key_to_filename = dict()
        self.multiscale = multiscale
        self.patch_shape = patch_shape
        self.percentiles = normalization_percentiles
        self.proposals = dict()
        self.transform = ImageTransforms() if transform else False

    # --- Load Data ---
    def ingest_proposals(
        self, brain_id, img_path, voxels, labels=None, paths=None
    ):
        """
        Ingests proposals represented by voxel coordinates along with optional
        labels and paths into the dataset.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the whole-brain dataset.
        img_path : str
            Prefix (or path) of a whole-brain image stored in a S3 bucket.
        voxels : List[Tuple[int]]
            Voxel coordinates representing the proposals.
        labels : List[int], optional
            Ground truth labels of the proposals. Default is None.
        paths : List[str], optional
            File paths corresponding to the proposal. Default is None.
        """
        # Sanity checks
        if labels is not None:
            assert len(voxels) == len(labels), "#proposals != #labels"
        if paths is not None:
            assert len(voxels) == len(paths), "#proposals != #paths"

        # Load image (if applicable)
        if brain_id not in self.imgs:
            self.imgs[brain_id] = img_util.TensorStoreImage(img_path)
            self.img_paths[brain_id] = img_path

        # Load proposal voxel coordinates
        for i, voxel in enumerate(voxels):
            key = (brain_id, tuple(voxel))
            self.proposals[key] = -1 if labels is None else labels[i]
            if paths is not None:
                self.key_to_filename[key] = paths[i]

    def ingest_proposals_from_df(self, df, img_pathes):
        """
        Iterates over a soma DataFrame and ingests proposals into a dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns: brain_id, label, swc_filename, x, y, z.
        img_pathes : Dict[str, str]
            Dictionary that maps a brain_id and returns the image prefix.
        """
        for brain_id, group in df.groupby("brain_id"):
            img_path = os.path.join(img_pathes[brain_id], str(self.multiscale))
            voxels = [img_util.to_voxels(xyz, self.multiscale) for xyz in group["xyz"]]
            labels = group["label"].tolist()
            paths = group["swc_filename"].tolist()

            self.ingest_proposals(
                brain_id=brain_id,
                img_path=img_path,
                voxels=voxels,
                labels=labels,
                paths=paths,
            )

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
        """
        _, img_patch, _ = self[key]
        img_util.plot_mips(img_patch)

    # --- Get Example ---
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

        Returns
        -------
        key : tuple
            Input "key" tuple.
        img_patch : numpy.ndarray
            3D image patch centered at "voxel".
        label : int
            Label associated with the proposal.
        """
        # Get voxel
        brain_id, voxel = key
        if self.transform:
            voxel = [u + random.randint(-5, 5) for u in voxel]

        # Get image patch
        img = self.get_patch(brain_id, voxel)
        if self.transform:
            img = self.transform(img)
        return key, img, self.proposals[key]

    def get_patch(self, brain_id, voxel):
        img = self.imgs[brain_id].read(voxel, self.patch_shape)
        img = np.minimum(img, self.brightness_clip)
        img = img_util.normalize(img, percentiles=self.percentiles)
        return img

    # --- Helpers ---
    def __len__(self):
        """
        Counts the number of proposals in self.

        Returns
        -------
        int
            Number of proposals in self.
        """
        return len(self.proposals)

    def n_positives(self):
        """
        Counts the number of positive proposals in the dataset.

        Returns
        -------
        int
            Number of positive proposals in the dataset.
        """
        return np.sum([1 for v in self.proposals.values() if v])

    def n_negatives(self):
        """
        Counts the number of negative proposals in the dataset.

        Returns
        -------
        int
            Number of negative proposals in the dataset.
        """
        return np.sum([1 for v in self.proposals.values() if not v])


# --- Custom Dataloader ---
class DataLoader:

    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle indices
        keys = list(self.dataset.proposals.keys())
        if self.shuffle:
            np.random.shuffle(keys)

        # Yield batches
        for i in range(0, len(keys), self.batch_size):
            batch_keys = keys[i: i + self.batch_size]
            keys_out, patches, labels = zip(*self.get_batch(batch_keys))
            patches = self.to_tensor(patches)
            labels = self.to_tensor(labels)
            yield list(keys_out), patches, labels

    # --- Helpers ---
    def __len__(self):
        """
        Counts number of examples in the dataset.

        Returns
        -------
        int
            Number of examples in the dataset.
        """
        return len(self.dataset)

    def get_batch(self, keys):
        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(self.dataset.__getitem__, k): k for k in keys}
            results = []
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"WARNING: {e} — skipping")
            return results

    @staticmethod
    def to_tensor(data):
        data = [torch.tensor(d, dtype=torch.float) for d in data]
        return torch.stack(data).unsqueeze(1)
