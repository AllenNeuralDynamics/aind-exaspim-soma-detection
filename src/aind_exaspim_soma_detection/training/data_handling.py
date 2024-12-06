"""
Created on Mon Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


"""

from torch.utils.data import Dataset

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
def fetch_smartsheet_somas(
    smartsheet_path,
    img_lookup_path,
    anisotropy=[1.0, 1.0, 1.0],
    multiscale=0,
):
    # Read data
    soma_coords = util.extract_somas_from_smartsheet(smartsheet_path)
    img_name_lookup = util.read_json(img_lookup_path)

    # Reformat data
    data = list()
    for brain_id, xyz_list in soma_coords.items():
        img_name = img_name_lookup[brain_id]
        img_path = f"s3://aind-open-data/{img_name}/fused.zarr/{multiscale}"
        voxels = [img_util.to_voxels(xyz, multiscale) for xyz in xyz_list]
        labels = len(soma_coords[brain_id]) * [1]
        data.append((brain_id, img_path, voxels, labels))
    return data


def fetch_soma_nonsoma_2024():
    pass
