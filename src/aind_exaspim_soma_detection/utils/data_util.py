"""
Created on Mon Jan 6 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading training data.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter

import ast
import os

from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import img_util, util


# --- Fetch Data ---
def load_examples(path):
    test_examples = list()
    for line in util.read_txt(path):
        idx = line.find(",")
        brain_id = ast.literal_eval(line[1:idx])
        xyz = ast.literal_eval(line[idx + 2: -1])
        test_examples.append((brain_id, xyz))
    return test_examples


def fetch_smartsheet_somas(
    smartsheet_path, img_prefixes_path, multiscale, adjust_coords_bool=True
):
    # Read data
    img_prefixes = util.read_json(img_prefixes_path)
    soma_coords = util.extract_somas_from_smartsheet(smartsheet_path)

    # Reformat data
    data = list()
    for brain_id, xyz_list in soma_coords.items():
        if brain_id not in ["686955", "708373"]:
            # Check whether to adjust coordinates
            if adjust_coords_bool:
                xyz_list = shift_somas(img_prefixes[brain_id], xyz_list)

            # Add examples
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
            - "img_path" (str): Path to image stored in S3 bucket.
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


# --- Adjust Smartsheet Coordinates ---
def shift_somas(img_prefix, xyz_list, multiscale=3, patch_shape=(40, 40, 40)):
    """
    Shifts soma coordinates from dendritic shaft to soma center.

    Parameters
    ----------
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    xyz_list : List[Tuples[float]
        List of soma coordinates to process.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        The default is 3.
    patch_shape : Tuple[int], optional
        Shape of the image patch to be extracted around each soma. The default
        is (40, 40, 40).

    Returns:
    --------
    List[Tuple[float]]
        Shifted soma coordinates in physical space.

    """
    img = img_util.open_img(img_prefix + str(multiscale))
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for xyz in xyz_list:
            threads.append(executor.submit(shift_soma, img, xyz, patch_shape))

        # Process results
        shifted_soma_xyz_list = list()
        for thread in as_completed(threads):
            shifted_xyz = thread.result()
            if shifted_xyz is not None:
                shifted_soma_xyz_list.append(shifted_xyz)
    return shifted_soma_xyz_list


def shift_soma(img, xyz, patch_shape, multiscale=3):
    """
    Shifts soma position from the beginning of the dendritic shaft to the
    center of the soma in a 3D image.

    Parameters
    ----------
    img : zarr.core.Array
        Array representing a 3D image of a whole brain.
    xyz : tuple or list of int
        Physical coordinate of soma.
    patch_shape : Tuple[int]
        Shape of the image patch to be extracted from "img".
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.

    Returns
    -------
    numpy.ndarray or None
        If a soma is detected, returns the adjusted soma center coordinates in
        physical space. If no valid shift is detected, it returns None.

    """
    voxel = img_util.to_voxels(xyz, multiscale=multiscale)
    img_patch = img_util.get_patch(img, voxel, patch_shape)
    shift = get_soma_shift(img_patch)
    if shift is not None:
        return img_util.local_to_physical(voxel, shift[::-1], multiscale)
    else:
        return None


def get_soma_shift(img_patch):
    """
    Detects soma center in a 3D image patch and computes a shift vector to
    adjust the soma's coordinates from the beginning of the dendritic shaft
    to the center of the soma.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image patch that contains a soma to be detected.

    Returns
    -------
    List[int] or None
        Shift vector to adjust soma coordinaute if a soma is detected. If no
        somas are detected, then returns None.

    """
    # Step 1: Generate Initial Proposals
    img_patch = gaussian_filter(img_patch, sigma=0.5)
    proposals_1 = spg.detect_blobs(img_patch, 140, 8, 12)
    proposals_2 = spg.detect_blobs(img_patch, 140, 6, 12)
    proposals_3 = spg.detect_blobs(img_patch, 140, 4, 12)
    proposals = proposals_1 + proposals_2 + proposals_3

    # Step 2: Filter Initial Proposals
    proposals = spg.spatial_filtering(proposals, 6)
    proposals = spg.gaussian_fitness_filtering(
        img_patch, proposals, min_score=0.7
    )
    proposals = spg.brightness_filtering(img_patch, proposals, 1)
    if len(proposals) > 0:
        shape = img_patch.shape
        return [s_i - ps_i // 2 for s_i, ps_i in zip(proposals[0], shape)]
    else:
        return None
