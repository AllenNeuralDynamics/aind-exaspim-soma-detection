"""
Created on Mon Jan 6 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading soma proposal data for testing and training.

"""

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import ast
import numpy as np
import pandas as pd
import os

from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import img_util, util


# --- Fetch Data ---
def fetch_smartsheet_somas(dataset_path, img_prefixes_path, multiscale):
    """
    Fetches and formats data from a text file generated from the Neuron
    Reconstruction SmartSheet.

    Parameters
    ----------
    dataset_path : str
        Path to the text file where each line is formatted as "(brain_id,
        voxel)".
    img_prefixes_path : str
        Path to a JSON file containing image prefixes for each brain ID.
    multiscale : int
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
    for brain_id, voxel in load_examples(dataset_path):
        data.append(
            reformat_data(brain_id, img_prefixes, multiscale, [voxel], 1)
        )
    return data


def fetch_exaspim_somas_2024(dataset_path, img_prefixes_path, multiscale):
    """
    Fetches and formats data from exaSPIM datasets.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory containing brain-specific subdirectories
        with "accepts" and "rejects" folders.
    img_prefixes_path : str
        Path to a JSON file containing image prefixes for each brain ID.
    multiscale : int
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
        # Accepts
        accepts_dir = os.path.join(dataset_path, brain_id, "accepts")
        data.append(
            load_swc_examples(
                accepts_dir, brain_id, img_prefixes, multiscale, 1
            )
        )

        # Rejects
        rejects_dir = os.path.join(dataset_path, brain_id, "rejects")
        data.append(
            load_swc_examples(
                rejects_dir, brain_id, img_prefixes, multiscale, 0
            )
        )
    return data


def load_swc_examples(swc_dir, brain_id, img_prefixes, multiscale, label):
    """
    Loads SWC files, converts soma coordinates to voxel format, and reformats
    the data for training and testing.

    Parameters
    ----------
    swc_dir : str
        Directory containing SWC files to be read.
    brain_id : str
        Unique identifier for the whole-brain dataset.
    img_prefixes : dict
        Dictionary that maps brain IDs to image S3 prefixes.
    multiscale : int
        Level in the image pyramid that the voxel coordinates must index into.
    label : int
        Label with each SWC file.

    Returns
    -------
    List[tuple]
        List of tuples where each tuple contains the following:
            - "brain_id" (str): Unique identifier for the brain.
            - "img_path" (str): Path to image stored in S3 bucket.
            - "voxels" (list): Voxel coordinates of proposed somas.
            - "labels" (list): Labels corresponding to voxels.

    """
    paths, xyz_list = util.read_swc_dir(swc_dir)
    voxels = [img_util.to_voxels(xyz, multiscale) for xyz in xyz_list]
    data = reformat_data(
        brain_id, img_prefixes, multiscale, voxels, label, paths
    )
    return data


def reformat_data(
    brain_id, img_prefixes, multiscale, voxels, label, paths=None
):
    """
    Reformats data for training or inference by converting xyz to voxel
    coordinates and associates them with a brain id, image path, and labels.

    Parameters
    ----------
    brain_id : str
        Unique identifier for the whole-brain dataset.
    img_prefixes : dict
        Dictionary that maps brain IDs to image S3 prefixes.
    multiscale : int
        Level in the image pyramid that the voxel coordinates must index into.
    voxels : List[ArrayLike]
        List of voxel coordinates.
    label : int
        Label associated with the given coordinates (i.e. 1 for "accepts" and
        0 for "rejects").
    paths : List[str], optional
        List of file paths corresponding to the examples in xyz_list. The
        default is None.

    Returns
    -------
    tuple
        Tuple that contains the following:
            - "brain_id" (str): Unique identifier for the brain.
            - "img_path" (str): Path to image stored in S3 bucket.
            - "voxels" (list): Voxel coordinates of proposed somas.
            - "labels" (list): Labels corresponding to voxels.

    """
    img_path = img_prefixes[brain_id] + str(multiscale)
    labels = len(voxels) * [label]
    if paths is None:
        return (brain_id, img_path, voxels, labels)
    else:
        return (brain_id, img_path, voxels, labels, paths)


def load_examples(path):
    """
    Loads examples stored in a text file where each line is formatted as
    "(brain_id, voxel)".

    Parameters
    ----------
    path : str
        Path to text file to be parsed.

    Returns
    -------
    List[Tuple[str, ArrayLike]]
        List of tuples such that each contains a "brain_id" and "voxel"
        coordinate.

    """
    examples = list()
    for line in util.read_txt(path):
        idx = line.find(",")
        brain_id = ast.literal_eval(line[1:idx])
        voxel = ast.literal_eval(line[idx + 2: -1])
        examples.append((brain_id, voxel))
    return examples


# --- Read SmartSheet ---
def scrape_smartsheet(smartsheet_path, img_prefixes_path, multiscale):
    """
    Scrapes data from a Smartsheet containing soma xyz coordinates, shifts the
    coordinate to the center of the soma, and reformats the data.

    Parameters
    ----------
    smartsheet_path : str
        Path to the Smartsheet file containing soma coordinates.
    img_prefixes_path : str
        Path to a JSON file containing image prefixes for each brain ID.
    multiscale : int
        Level in the image pyramid that the voxel coordinates must index into.

    Returns
    -------
    List[tuple]
        List of tuples containing processed soma data for each brain.

    """
    # Read data
    img_prefixes = util.read_json(img_prefixes_path)
    soma_coords = extract_smartsheet_somas(smartsheet_path)

    # Center somas and reformat data
    data = list()
    with ProcessPoolExecutor() as executor:
        # Assign processes
        processes = list()
        for brain_id, xyz_list in soma_coords.items():
            if brain_id not in ["686955", "708373"]:
                processes.append(
                    executor.submit(
                        shift_somas, brain_id, img_prefixes[brain_id], xyz_list
                    )
                )

        # Store results
        for process in tqdm(as_completed(processes), total=len(processes)):
            brain_id, xyz_list = process.result()
            data.append(
                reformat_data(brain_id, img_prefixes, multiscale, xyz_list, 1)
            )
    return data


def extract_smartsheet_somas(path, soma_status=None):
    """
    Extracts soma coordinates from the AIND neuron reconstructions Smartsheet
    which is assumed to be stored locally as an Excel file.

    Parameters
    ----------
    path : str
        Path to the Smartsheet Excel file. Note that this file must contain a
        sheet named "Neuron Reconstructions".
    soma_status : str, optional
        Specifies the filter condition for somas based on their status. If
        provided, filters soma coordinates that match the specified status
        (case-insensitive). The default is None, which includes all soma

    Returns
    -------
    Dict[(str, list)]
        Dictionary where the keys are brain IDs and values are lists of soma
        coordinates extracted from the sheet.

    """
    # Initializations
    df = pd.read_excel(path, sheet_name="Neuron Reconstructions")
    n_somas = 0
    somas = dict()
    if type(soma_status) is str:
        soma_status = soma_status.lower()

    # Parse dataframe
    idx = 0
    while idx < len(df["Collection"]):
        microscope = df["Collection"][idx]
        if type(microscope) is str:
            brain_id = str(df["ID"][idx])
            if "spim" in microscope.lower() and brain_id != "609281":
                somas[brain_id] = get_soma_coords(df, idx + 1, soma_status)
                n_somas += len(somas[brain_id])
        idx += 1
    return somas


def get_soma_coords(df, idx, soma_status):
    """
    Extracts a list of 3D coordinates of soma from a DataFrame starting at a
    specified index.

    Parameters
    -----------
    df : pandas.DataFrame
        DataFrame containing a column of soma coordinates.
    idx : int
        Index in the DataFrame where the soma coordinates start.
    soma_status : str or None
        ...

    Returns
    --------
    numpy.ndarray
        Array of 3D coordinates.

    """
    xyz_list = list()
    while type(df["Horta Coordinates"][idx]) is str:
        item = df["Horta Coordinates"][idx]
        if "[" in item and "]" in item:
            # Check status
            is_status_str = type(df["Status 1"][idx]) is str
            if soma_status is not None and is_status_str:
                if df["Status 1"][idx].lower() not in soma_status:
                    idx += 1
                    continue

            # Add coordinate
            xyz_list.append(tuple(ast.literal_eval(item)))
            assert len(xyz_list[-1]) == 3, "Coordinate is not 3D!"
        idx += 1
        if idx >= len(df["Horta Coordinates"]):
            break
    return np.array(xyz_list)


# --- Adjust Smartsheet Coordinates ---
def shift_somas(
    brain_id, img_prefix, xyz_list, multiscale=3, patch_shape=(36, 36, 36)
):
    """
    Shifts soma coordinates from dendritic shaft to soma center.

    Parameters
    ----------
    brain_id : str
        Unique identifier for the whole-brain dataset.
    img_prefix : str
        Prefix (or path) of a whole-brain image stored in a S3 bucket.
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
    str, List[Tuple[float]]
        Brain id and shifted soma xyz coordinates.

    """
    img = img_util.open_img(img_prefix + str(multiscale))
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for xyz in xyz_list:
            threads.append(executor.submit(shift_soma, img, xyz, patch_shape))

        # Process results
        shifted_xyz_list = list()
        for thread in as_completed(threads):
            shifted_xyz = thread.result()
            if shifted_xyz is not None:
                shifted_xyz_list.append(shifted_xyz)

    return brain_id, shifted_xyz_list


def shift_soma(img, xyz, patch_shape, multiscale=3):
    """
    Shifts soma position from the beginning of the dendritic shaft to the
    center of the soma in a 3D image.

    Parameters
    ----------
    img : zarr.core.Array
        Array representing a 3D image of a whole-brain.
    xyz : tuple or list of int
        xyz coordinate of soma.
    patch_shape : Tuple[int]
        Shape of the image patch to be extracted from "img".
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.

    Returns
    -------
    numpy.ndarray or None
        If a soma is detected, returns the adjusted soma xyz coordinates. If
        no soma is detected, it returns None.

    """
    voxel = img_util.to_voxels(xyz, multiscale=multiscale)
    img_patch = img_util.get_patch(img, voxel, patch_shape)
    shift = get_soma_shift(img_patch)
    if shift is not None:
        return img_util.local_to_physical(voxel, shift, multiscale)
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
    proposals = spg.brightness_filtering(img_patch, proposals, 5)
    proposals = spg.gaussian_fitness_filtering(
        img_patch, proposals, min_score=0.8
    )
    proposals = spg.brightness_filtering(img_patch, proposals, 1)
    if len(proposals) > 0:
        shape = img_patch.shape
        return [ps_i - s_i // 2 for s_i, ps_i in zip(shape, proposals[0])]
    else:
        return None
