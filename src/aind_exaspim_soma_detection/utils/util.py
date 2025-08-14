"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor

from io import StringIO
from random import sample
from zipfile import ZipFile

import boto3
import json
import numpy as np
import os
import shutil


# --- OS utils ---
def mkdir(path, delete=False):
    """
    Creates a directory located at the given path.

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. Default is False.
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
        Names of subdirectories.
    """
    subdir_names = list()
    for d in os.listdir(directory_path):
        path = os.path.join(directory_path, d)
        if os.path.isdir(path) and not d.startswith("."):
            subdir_names.append(d)
    return subdir_names


def list_paths(directory, extension=""):
    """
    Lists all paths within "directory" ending with "extension" if provided.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        If provided, only paths of files with the extension are returned.
        Default is an empty string.

    Returns
    -------
    List[str]
        Paths within "directory".
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
        Contents of a txt file.
    """
    with open(path, "r") as f:
        return f.read().splitlines()


def read_json(path):
    """
    Reads JSON file stored at "path".

    Parameters
    ----------
    path : str
        Path where JSON file is stored.

    Returns
    -------
    str
        Contents of JSON file.
    """
    with open(path, "r") as file:
        return json.load(file)


def write_json(path, my_dict):
    """
    Writes the contents in the given dictionary to a JSON file at "path".

    Parameters
    ----------
    path : str
        Path where JSON file is stored.
    my_dict : dict
        Dictionary to be written to a JSON.
    """
    with open(path, 'w') as file:
        json.dump(my_dict, file, indent=4)


def write_list(path, my_list):
    """
    Writes each item in a list to a text file, with each item on a new line.

    Parameters
    ----------
    path : str
        Path where text file is to be written.
    my_list
        Items to write to a text file.
    """
    with open(path, "w") as file:
        for item in my_list:
            file.write(f"{tuple(item)}\n")


# --- SWC utils ---
def read_swc_dir(swc_dir):
    """
    Reads all SWC files in a given directory and returns the content. Note
    that each SWC file is assumed to contain a single point.

    Parameters
    ----------
    swc_dir : str
        Path to the directory containing SWC files.

    Returns
    -------
    Tuple[list]
        Paths to SWC files and xyz coordinates read from corresponding SWC
        files.
    """
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = list()
        for path in list_paths(swc_dir, extension=".swc"):
            threads.append(executor.submit(read_swc, path))

        # Process results
        path_list = list()
        xyz_list = list()
        for thread in as_completed(threads):
            path, xyz = thread.result()
            path_list.append(path)
            xyz_list.append(xyz)
        return path_list, xyz_list


def read_swc(path):
    """
    Processes lines of text from a content source, extracts an offset value
    and returns the remaining content starting from the line immediately after
    the last commented line.

    Parameters
    ----------
    path : str
        Path to an SWC file.

    Returns
    -------
    List[float]
        xyz coordinate stored in SWC file.
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
    return path, [float(xyz_str[i]) + offset[i] for i in range(3)]


def write_points(zip_path, points, color=None, prefix="", radius=20):
    """
    Writes a list of 3D points to individual SWC files in the specified
    directory.

    Parameters
    -----------
    zip_path : str
        Path to ZIP archive where the SWC files will be saved.
    points : list
        A list of 3D points to be saved.
    color : str, optional
        The color to associate with the points in the SWC files. Default is
        None.
    prefix : str, optional
        String that is prefixed to the filenames of the SWC files. Default is
        an empty string. Default is an empty string.
    radius : float, optional
        Radius to be used in SWC file. Default is 20.
    """
    zip_writer = ZipFile(zip_path, "w")
    for i, xyz in enumerate(points):
        filename = prefix + str(i + 1) + ".swc"
        to_zipped_point(zip_writer, filename, xyz, color=color, radius=radius)


def to_zipped_point(zip_writer, filename, xyz, color=None, radius=5):
    """
    Writes a point to an SWC file format, which is then stored in a ZIP
    archive.

    Parameters
    ----------
    zip_writer : zipfile.ZipFile
        A ZipFile object that will store the generated SWC file.
    filename : str
        Filename of SWC file.
    xyz : ArrayLike
        Point to be written to SWC file.
    color : str, optional
        Color of nodes. Default is None.
    radius : float, optional
        Radius of point. Default is 5um.
    """
    with StringIO() as text_buffer:
        # Preamble
        if color:
            text_buffer.write("# COLOR " + color)
        text_buffer.write("\n" + "# id, type, z, y, x, r, pid")

        # Write entry
        x, y, z = tuple(xyz)
        text_buffer.write("\n" + f"1 5 {x} {y} {z} {radius} -1")

        # Finish
        zip_writer.writestr(filename, text_buffer.getvalue())


# --- S3 utils ---
def upload_dir_to_s3(bucket_name, prefix, source_dir):
    """
    Uploads the contents of a directory to S3.

    Parameters
    ----------
    bucket_name : str
        Name of S3 bucket.
    prefix : str
        Name of S3 prefix to be written to.
    source_dir : str
        Path to local directory to be written to S3.
    """
    for name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, name)
        if os.path.isdir(source_path):
            subprefix = os.path.join(prefix, name)
            upload_dir_to_s3(bucket_name, subprefix, source_path)
        else:
            destination_path = os.path.join(prefix, name)
            upload_file_to_s3(bucket_name, source_path, destination_path)
    print("Results uploaded to", f"s3://{bucket_name}/{prefix}")


def upload_file_to_s3(bucket_name, source_path, destination_path):
    """
    Uploads single file on local machine to S3.

    Parameters
    ----------
    bucket_name : str
        Name of S3 bucket.
    source_path : str
        Path to local file to be written to S3.
    destination_path : str
        Path within S3 bucket that file is to be written to.
    """
    s3 = boto3.client("s3")
    s3.upload_file(source_path, bucket_name, destination_path)


# --- Miscellaneous ---
def compute_std(values, weights=None):
    """
    Compute weighted standard deviation.

    Parameters
    ----------
    values : array-like
        Data points.
    weights : array-like, optional
        Weights corresponding to each data point. Default is None.

    Returns
    -------
    float
        Weighted standard deviation.
    """
    weights = weights or np.ones_like(values)
    values = np.array(values)
    weights = np.array(weights)

    weighted_mean = np.sum(weights * values) / np.sum(weights)
    variance = np.sum(weights * (values - weighted_mean)**2) / np.sum(weights)
    return np.sqrt(variance)


def find_key_intersection(dict_1, dict_2):
    keys_1 = find_key_subset(dict_1, dict_2)
    keys_2 = find_key_subset(dict_2, dict_1)
    return keys_1.intersection(keys_2)


def find_key_subset(dict_1, dict_2):
    subset = set()
    for key in dict_1:
        if key in dict_2:
            subset.add(key)
    return subset


def get_subdict(my_dict, keys):
    subdict = dict()
    for key in keys:
        subdict[key] = my_dict[key]
    return subdict


def sample_once(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    any
    """
    return sample(my_container, 1)[0]


def time_writer(t, unit="seconds"):
    """
    Converts a runtime "t" to a larger unit of time if applicable.

    Parameters
    ----------
    t : float
        Runtime.
    unit : str, optional
        Unit of time that "t" is expressed in. Default is "seconds".

    Returns
    -------
    float
        Runtime
    str
        Unit of time.
    """
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit
