"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import boto3
import numpy as np
import pandas as pd


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


# --- Extracting somas from smartsheets ---
def extract_somas_from_smartsheet(path, soma_status=None):
    # Initializations
    df = pd.read_excel(path, sheet_name="Neuron Reconstructions")
    n_somas = 0
    soma_coords = dict()
    if type(soma_status) is str:
        soma_status = soma_status.lower()

    # Parse dataframe
    idx = 0
    while idx < len(df["Collection"]):
        if type(df["Collection"][idx]) is str:
            if "spim" in df["Collection"][idx].lower():
                brain_id = df["ID"][idx]
                soma_coords[brain_id] = get_soma_coords(df, idx + 1, soma_status)
                n_somas += len(soma_coords[brain_id])
        idx += 1

    # Report Results
    print("# Whole Brain Samples:", len(soma_coords))
    print("# Somas:", n_somas)
    return soma_coords


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
            xyz_list.append(xyz_from_str(item))
            assert len(xyz_list[-1]) == 3, "Coordinate is not 3D!"
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
                executor.submit(save_point, path, xyz, 20, color=color)
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
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter="/")
    subprefixes = []
    if "CommonPrefixes" in response:
        subprefixes = [cp["Prefix"] for cp in response["CommonPrefixes"]]
    return subprefixes


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
    prefixes = []
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


def find_img_prefix(bucket_name, prefixes, sample_id):
    cur_prefix = None
    cur_date = None
    for prefix in prefixes:
        # Check if prefix is valid
        if str(sample_id) in prefix and "fusion" in prefix:
            # Check if prefix has fused image
            if is_file_in_prefix(bucket_name, prefix, "fused.zarr"):
                # Check if prefix is most recent
                date = get_upload_date(prefix)
                if cur_prefix is None:
                    cur_prefix = prefix
                    cur_date = date
                elif date == most_recent_date(date, cur_date):
                    cur_prefix = prefix
                    cur_date = date
    return cur_prefix + "fused.zarr/" if cur_prefix is not None else cur_prefix


# --- Miscellaneous ---
def get_upload_date(prefix):
    """
    Extracts the upload date from an S3 prefix string.

    Parameters
    -----------
    prefix : str
        S3 prefix (path) containing the upload date in the format
        "<name>/fusion_<date>/".

    Returns
    --------
    str
        Extracted date string without slashes.

    """
    date = prefix.split("fusion_")[-1]
    return date.replace("/", "")


def most_recent_date(date1_str, date2_str, date_format="%Y-%m-%d_%H-%M-%S"):
    """
    Compares two date strings and determines which one is more recent.

    Parameters
    -----------
    date1_str : str
        First date string to compare.
    date2_str : str
        Second date string to compare.
    
    date_format : str, optional
        The format in which the dates are provided. The default is
        "%Y-%m-%d %H:%M:%S".
    
    Returns
    --------
    str
       More recent data in the same given format.

    """
    try:
        date1 = datetime.strptime(date1_str, date_format)
        date2 = datetime.strptime(date2_str, date_format)
        return date1_str if date1 > date2 else date2_str
    except ValueError:
        return ""
