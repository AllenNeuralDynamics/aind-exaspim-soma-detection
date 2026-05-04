"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from botocore import UNSIGNED
from botocore.client import Config
from google.cloud import storage
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
    paths : List[str]
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
    Reads txt file at the given path.

    Parameters
    ----------
    path : str
        Path to txt file.

    Returns
    -------
    List[str]
        Lines from the txt file.
    """
    if is_s3_path(path):
        return read_txt_from_s3(path)
    elif is_gcs_path(path):
        return read_txt_from_gcs(path)
    else:
        with open(path, "r") as f:
            return f.read()


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
    with open(path, "w") as file:
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


# --- SWC Utils ---
def write_points(zip_path, points, color=None, prefix="", radius=20):
    """
    Writes a list of 3D points to individual SWC files in the specified
    directory.

    Parameters
    -----------
    zip_path : str
        Path to ZIP archive where the SWC files will be saved.
    points : list
        3D points to be saved.
    color : str, optional
        The color to associate with the points in the SWC files. Default is
        None.
    prefix : str, optional
        String that is prefixed to the filenames of the SWC files. Default is
        an empty string.
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
        text_buffer.write("\n# id, type, z, y, x, r, pid")

        # Write entry
        x, y, z = tuple(xyz)
        text_buffer.write(f"\n1 5 {x} {y} {z} {radius} -1")
        zip_writer.writestr(filename, text_buffer.getvalue())


# --- GCS Utils ---
def is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


def list_gcs_subdirs(bucket_name, prefix):
    """
    Lists all direct subdirectories of a given prefix in a GCS bucket.

    Parameters
    ----------
    bucket : str
        Name of bucket to be read from.
    prefix : str
        Path to directory in "bucket".

    Returns
    -------
    subdirs: List[str]
         Direct subdirectories.
    """
    prefix = prefix.rstrip("/") + "/"
    subdirs = set()
    bucket = storage.Client().bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        remainder = blob.name[len(prefix):]
        subdir = remainder.split("/")[0]
        if subdir:
            subdirs.add(subdir)
    return sorted(subdirs)


def read_txt_from_gcs(path):
    """
    Reads a txt file stored in a GCS bucket.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subpath = parse_cloud_path(path)
    bucket = storage.Client().bucket(bucket_name)
    return bucket.blob(subpath).download_as_text()


# --- S3 utils ---
def is_s3_path(path):
    """
    Checks if the given path is an S3 path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


def read_txt_from_s3(path):
    """
    Reads a txt file stored in an S3 bucket.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    bucket_name, subpath = parse_cloud_path(path)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    obj = s3.get_object(Bucket=bucket_name, Key=subpath)
    return obj["Body"].read().decode("utf-8")


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
    variance = np.sum(weights * (values - weighted_mean) ** 2) / np.sum(
        weights
    )
    return np.sqrt(variance)


def find_key_intersection(dict_1, dict_2):
    """
    Find the set of keys that exist in both dictionaries.

    Parameters
    ----------
    dict_1 : dict
        First dictionary to compare.
    dict_2 : dict
        Second dictionary to compare.

    Returns
    -------
    set
        A set containing the keys that are present in both dictionaries.
    """
    keys_1 = find_key_subset(dict_1, dict_2)
    keys_2 = find_key_subset(dict_2, dict_1)
    return keys_1.intersection(keys_2)


def find_key_subset(dict_1, dict_2):
    """
    Find the subset of keys from the first dictionary that also exist in the
    second.

    Parameters
    ----------
    dict_1 : dict
        Dictionary whose keys will be checked.
    dict_2 : dict
        Dictionary against which membership of keys is tested.

    Returns
    -------
    subset : set
        A set containing the keys from "dict_1" that are also present in
        "dict_2".
    """
    subset = set()
    for key in dict_1:
        if key in dict_2:
            subset.add(key)
    return subset


def get_subdict(my_dict, keys):
    """
    Extract a subdictionary containing only the specified keys.

    Parameters
    ----------
    my_dict : dict
        The source dictionary to extract values from.
    keys : iterable
        An iterable of keys to include in the resulting dictionary.

    Returns
    -------
    dict
        A dictionary containing only the specified keys from "my_dict".
    """
    subdict = dict()
    for key in keys:
        subdict[key] = my_dict[key]
    return subdict


def parse_cloud_path(path):
    """
    Parses a cloud storage path into its bucket name and prefix. Supports
    paths of the form: "{scheme}://bucket_name/prefix" or without a scheme.

    Parameters
    ----------
    path : str
        Path to be parsed.

    Returns
    -------
    bucket_name : str
        Name of the bucket.
    prefix : str
        Cloud prefix.
    """
    # Split path
    path = path[len("s3://"):] if is_s3_path else path[len("gs://"):]
    parts = path.split("/", 1)

    # Extract bucket and prefix
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


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
    t : float
        Runtime
    unit : str
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
