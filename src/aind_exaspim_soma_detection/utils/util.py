"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd


def mkdir(path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. The default is False.

    Returns
    -------
    None

    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes directory and all subdirectories at "path".

    Parameters
    ----------
    path : str
        Path to directory and subdirectories to be deleted if they exist.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        shutil.rmtree(path)


def write_to_s3(local_path, bucket_name, prefix):
    """
    Writes a single file on local machine to an s3 bucket.

    Parameters
    ----------
    local_path : str
        Path to file to be written to s3.
    bucket_name : str
        Name of s3 bucket.
    prefix : str
        Path within s3 bucket.

    Returns
    -------
    None

    """
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket_name, prefix)


# --- Extracting somas from smartsheets ---
def extract_somas_from_smartsheet(path):
    # Initializations
    df = pd.read_excel(path, sheet_name="Neuron Reconstructions")
    n_somas = 0
    soma_coords = dict()

    # Parse dataframe
    idx = 0
    while idx < len(df["Collection"]):
        if type(df["Collection"][idx]) is str:
            if "spim" in df["Collection"][idx].lower():
                brain_id = df["ID"][idx]
                soma_coords[brain_id] = get_soma_coordinates(df, idx + 1)
                n_somas += len(soma_coords[brain_id])
        idx += 1

    # Report Results
    print("# Whole Brain Samples:", len(soma_coords))
    print("# Somas:", n_somas)
    return soma_coords


def get_soma_coordinates(df, idx):
    """
    Extracts a list of 3D coordinates of soma from a DataFrame starting at a
    specified index.

    Parameters
    -----------
    df : pandas.DataFrame
        DataFrame containing a column of soma coordinates.
    idx : int
        Index in the DataFrame where the soma coordinates start.

    Returns
    --------
    numpy.ndarray
        Array of 3D coordinates.

    """
    # May want to add condition that soma must have been completed
    xyz_list = list()
    while type(df["Horta Coordinates"][idx]) is str:
        item = df["Horta Coordinates"][idx]
        if "[" in item and "]" in item:
            xyz = xyz_from_str(item)
            xyz_list.append(xyz)
            assert len(xyz) == 3, "Coordinate is not 3D!"
        idx += 1
    return np.array(xyz_list)


def xyz_from_str(xyz_str):
    """
    Converts a string representation of 3D coordinates into a list of floats.

    Parameters
    -----------
    xyz_str : str
        A string containing 3D coordinates in the format "[x, y, z]". Square
        brackets are optional, but values must be comma-separated.

    Returns
    --------
    Tuple[float]
        3D Coordinate from the given string.

    """
    xyz_str = xyz_str.replace("[", "")
    xyz_str = xyz_str.replace("]", "")
    return tuple([float(x) for x in xyz_str.split(",")])


# --- swc utils ---
def write_points(output_dir, points, color=None, prefix=""):
    """
    Writes a list of 3D points to individual SWC files in the specified
    directory.

    Parameters
    -----------
    output_dir : str
        Directory where the SWC files will be saved.
    points : list
        A list of 3D points to be saved.
    color : str, optional
        The color to associate with the points in the SWC files. The default
        is None.
    prefix : str, optional
        String that is prefixed to the filenames of the SWC files. Default is
        an empty string.

    Returns
    --------
    None

    """
    mkdir(output_dir, delete=True)
    with ThreadPoolExecutor() as executor:
        # Assign Threads
        threads = list()
        for i, xyz in enumerate(points):
            filename = prefix + str(i + 1) + ".swc"
            path = os.path.join(output_dir, filename)
            threads.append(
                executor.submit(save_point, path, xyz, 10, color=color)
            )


def save_point(path, xyz, radius=5, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    xyz : ArrayLike
        xyz coordinate to be written to an swc file.
    radius : float, optional
        Radius of point. The default is 5um.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        # Preamble
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, x, y, z, r, pid")
        f.write("\n")

        # Entry
        x, y, z = tuple(xyz)
        f.write(f"1 5 {x} {y} {z} {radius} -1")
