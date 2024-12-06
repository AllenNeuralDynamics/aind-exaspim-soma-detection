"""
Created on Mon Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

A custom dataset used to train a neural network for classifying soma proposals
as either accepted or rejected. The dataset is initialized by providing the
following inputs:
    (1) Path to a whole brain dataset stored in an S3 bucket
    (2) List of voxel coordinates representing soma proposals
    (3) optionally, labels for each proposal (i.e. accept or reject)

Note: This dataset supports these inputs from multiple whole brain datasets.

"""

from torch.utils.data import Dataset

from aind_exaspim_soma_detection.utils import img_util, util


class SomaDataset(Dataset):
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
