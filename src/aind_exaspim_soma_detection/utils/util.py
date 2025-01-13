"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from concurrent.futures import ThreadPoolExecutor
from random import sample

import ast
import boto3
import json
import numpy as np
import os
import pandas as pd
import shutil


# --- os utils ---
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


def list_subdirectory_names(directory_path):
    """
    Lists the names of all subdirectories in the given directory.

    Parameters
    ----------
    directory_path : str
        Path to the directory to search.

    Returns
    -------
    List[str]
        List of the names of subdirectories.

    """
    subdir_names = list()
    for d in os.listdir(directory_path):
        path = os.path.join(directory_path, d)
        if os.path.isdir(path) and not d.startswith("."):
            subdir_names.append(d)
    return subdir_names


def list_paths(directory, extension=""):
    """
    Lists all paths within "directory" that end with "extension" if provided.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        If provided, only paths of files with the extension are returned. The
        default is an empty string.

    Returns
    -------
    list[str]
        List of all paths within "directory".

    """
    paths = list()
    for f in os.listdir(directory):
        if f.endswith(extension):
            paths.append(os.path.join(directory, f))
    return paths


# --- IO utils ---
def read_txt(path):
    """
    Reads txt file stored at "path".

    Parameters
    ----------
    path : str
        Path where txt file is stored.

    Returns
    -------
    str
        Contents of txt file.

    """
    with open(path, "r") as f:
        return f.read().splitlines()


def read_json(path):
    """
    Reads json file stored at "path".

    Parameters
    ----------
    path : str
        Path where json file is stored.

    Returns
    -------
    str
        Contents of json file.

    """
    with open(path, "r") as file:
        return json.load(file)


def write_list_to_file(path, my_list):
    """
    Writes each item in a list to a text file, with each item on a new line.

    Parameters
    ----------
    path : str
        Path where text file is to be written.
    my_list
        The list of items to write to the file.

    Returns
    -------
    None

    """
    with open(path, "w") as file:
        for item in my_list:
            file.write(f"{item}\n")


# --- Extract Smartsheet Somas ---
def extract_somas_from_smartsheet(path, soma_status=None):
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
    return np.array(xyz_list)


# --- swc utils ---
def read_swc_dir(swc_dir):
    """
    Reads all swc files in a given directory and returns the content. Note
    that each swc file is assumed to contain a single point.

    Parameters
    ----------
    swc_dir : str
        Path to the directory containing swc files.

    Returns
    -------
    list
        xyz coordinates read from swc files.

    """
    return [read_swc(path) for path in list_paths(swc_dir, extension=".swc")]


def read_swc(path):
    """
    Processes lines of text from a content source, extracts an offset value
    and returns the remaining content starting from the line immediately after
    the last commented line.

    Parameters
    ----------
    path : str
        Path to an swc file.

    Returns
    -------
    List[float]
        xyz coordinate stored in swc file.

    """
    # Parse commented section
    offset = [0.0, 0.0, 0.0]
    for i, line in enumerate(read_txt(path)):
        if line.startswith("# OFFSET"):
            offset = [float(val) for val in line.split()[2:5]]
        if not line.startswith("#"):
            break

    # Extract xyz coordinate
    xyz_str = line.split()[2:5]
    return [float(xyz_str[i]) + offset[i] for i in range(3)]


def write_points(output_dir, points, color=None, prefix="", radius=20):
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
    radius : float, optional
        Radius to be used in swc file.

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
                executor.submit(write_point, path, xyz, radius, color=color)
            )


def write_point(path, xyz, radius=5, color=None):
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


# --- S3 utils ---
def list_s3_prefixes(bucket_name, prefix):
    """
    Lists all immediate subdirectories of a given S3 path (prefix).

    Parameters
    -----------
    bucket_name : str
        Name of the S3 bucket to search.
    prefix : str
        S3 prefix (path) to search within.

    Returns:
    --------
    List[str]
        List of immediate subdirectories under the specified prefix.

    """
    # Check prefix is valid
    if not prefix.endswith("/"):
        prefix += "/"

    # Call the list_objects_v2 API
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=bucket_name, Prefix=prefix, Delimiter="/"
    )
    if "CommonPrefixes" in response:
        return [cp["Prefix"] for cp in response["CommonPrefixes"]]
    else:
        return list()


def list_s3_bucket_prefixes(bucket_name, keyword=None):
    """
    Lists all top-level prefixes (directories) in an S3 bucket, optionally
    filtering by a keyword.

    Parameters
    -----------
    bucket_name : str
        Name of the S3 bucket to search.
    keyword : str, optional
        Keyword used to filter the prefixes.

    Returns
    --------
    List[str]
        A list of top-level prefixes (directories) in the S3 bucket. If a
        keyword is provided, only the matching prefixes are returned.

    """
    # Initializations
    prefixes = list()
    continuation_token = None
    s3 = boto3.client("s3")

    # Main
    while True:
        # Call the list_objects_v2 API
        list_kwargs = {"Bucket": bucket_name, "Delimiter": "/"}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)

        # Collect the top-level prefixes
        if "CommonPrefixes" in response:
            for prefix in response["CommonPrefixes"]:
                if keyword and keyword in prefix["Prefix"].lower():
                    prefixes.append(prefix["Prefix"])
                elif keyword is None:
                    prefixes.append(prefix["Prefix"])

        # Check if there are more pages to fetch
        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break
    return prefixes


def is_file_in_prefix(bucket_name, prefix, filename):
    """
    Checks if a specific file exists within a given S3 prefix.

    Parameters
    -----------
    bucket_name : str
        Name of the S3 bucket to searched.
    prefix : str
        S3 prefix (path) under which to look for the file.
    filename : str
        Name of the file to search for within the specified prefix.

    Returns:
    --------
    bool
        Returns "True" if the file exists within the given prefix,
        otherwise "False".

    """
    for sub_prefix in list_s3_prefixes(bucket_name, prefix):
        if filename in sub_prefix:
            return True
    return False


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


# --- Miscellaneous ---
def sample_once(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    sample

    """
    return sample(my_container, 1)[0]
