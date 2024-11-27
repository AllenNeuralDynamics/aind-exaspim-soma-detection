"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General helper routines for various tasks.

"""

import json
import math
import os
import shutil
from io import BytesIO
from random import sample
from zipfile import ZipFile

import boto3
import networkx as nx
import numpy as np
import psutil
from google.cloud import storage

from deep_neurographs.utils import graph_util as gutil


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


def listdir(path, extension=None):
    """
    Lists all files in the directory at "path". If an extension "ext" is
    provided, then only files containing "extension" are returned.

    Parameters
    ----------
    path : str
        Path to directory to be searched.
    extension : str, optional
       Extension of file type of interest. The default is None.

    Returns
    -------
    list
        Files in directory at "path" with extension "extension" if provided.
        Otherwise, list of all files in directory.

    """
    if extension is None:
        return [f for f in os.listdir(path)]
    else:
        return [f for f in os.listdir(path) if f.endswith(extension)]


def list_subdirs(path, keyword=None, return_paths=False):
    """
    Creates list of all subdirectories at "path". If "keyword" is provided,
    then only subdirectories containing "keyword" are contained in list.

    Parameters
    ----------
    path : str
        Path to directory containing subdirectories to be listed.
    keyword : str, optional
        Only subdirectories containing "keyword" are contained in list that is
        returned. The default is None.
    return_paths : bool
        Indication of whether to return full path of subdirectories. The
        default is False.

    Returns
    -------
    list
        List of all subdirectories at "path".

    """
    subdirs = list()
    for subdir in os.listdir(path):
        is_dir = os.path.isdir(os.path.join(path, subdir))
        is_hidden = subdir.startswith('.')
        if is_dir and not is_hidden:
            subdir = os.path.join(path, subdir) if return_paths else subdir
            if (keyword and keyword in subdir) or not keyword:
                subdirs.append(subdir)
    return sorted(subdirs)


def list_paths(directory, extension=None):
    """
    Lists all paths within "directory" that end with "extension" if provided.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        If provided, only paths of files with the extension are returned. The
        default is None.

    Returns
    -------
    list[str]
        List of all paths within "directory".

    """
    paths = list()
    for f in listdir(directory, extension=extension):
        paths.append(os.path.join(directory, f))
    return paths


def set_path(dir_name, filename, extension):
    """
    Sets the path for a file in a directory. If a file with the same name
    exists, then this routine finds a suffix to append to the filename.

    Parameters
    ----------
    dir_name : str
        Name of directory that path will be generated to point to.
    filename : str
        Name of file that path will contain.
    extension : str
        Extension of file.

    Returns
    -------
    str
        Path to file in "dirname" with the name "filename" and possibly some
        suffix.

    """
    cnt = 0
    extension = extension.replace(".", "")
    path = os.path.join(dir_name, f"{filename}.{extension}")
    while os.path.exists(path):
        path = os.path.join(dir_name, f"{filename}.{cnt}.{extension}")
        cnt += 1
    return path


# -- gcs utils --
def list_files_in_zip(zip_content):
    """
    Lists all files in a zip file stored in a GCS bucket.

    Parameters
    ----------
    zip_content : str
        Content stored in a zip file in the form of a string of bytes.

    Returns
    -------
    list[str]
        List of filenames in a zip file.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_gcs_filenames(bucket, prefix, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    bucket : google.cloud.client
        Name of bucket to be read from.
    prefix : str
        Path to directory in "bucket".
    extension : str
        File extension of filenames to be listed.

    Returns
    -------
    list
        Filenames stored at "cloud" path with the given extension.

    """
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if extension in blob.name]


def list_gcs_subdirectories(bucket_name, prefix):
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
    list[str]
         List of direct subdirectories.

    """
    # Load blobs
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs


# -- io utils --
def read_json(path):
    """
    Reads json file stored at "path".

    Parameters
    ----------
    path : str
        Path where json file is stored.

    Returns
    -------
    dict
        Contents of json file.

    """
    with open(path, "r") as f:
        return json.load(f)


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


def read_metadata(path):
    """
    Parses metadata file to extract the "chunk_origin" and "chunk_shape".

    Parameters
    ----------
    path : str
        Path to metadata file to be read.

    Returns
    -------
    list, list
        Chunk origin and chunk shape specified by metadata.

    """
    metadata = read_json(path)
    return metadata["chunk_origin"], metadata["chunk_shape"]


def read_zip(zip_file, path):
    """
    Reads the content of an swc file from a zip file.

    Parameters
    ----------
    zip_file : ZipFile
        Zip containing text file to be read.

    Returns
    -------
    str
        Contents of a txt file.

    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


def write_json(path, contents):
    """
    Writes "contents" to a json file at "path".

    Parameters
    ----------
    path : str
        Path that .txt file is written to.
    contents : dict
        Contents to be written to json file.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        json.dump(contents, f)


def write_txt(path, contents):
    """
    Writes "contents" to a txt file at "path".

    Parameters
    ----------
    path : str
        Path that txt file is written to.
    contents : dict
        Contents to be written to txt file.

    Returns
    -------
    None

    """
    f = open(path, "w")
    f.write(contents)
    f.close()


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
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket_name, prefix)


# --- math utils ---
def get_avg_std(data, weights=None):
    """
    Computes the average and standard deviation of "data". If "weights" is
    provided, the weighted average and standard deviation are computed.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be evaluated.
    weights : numpy.ndarray, optional
        Weights to apply to each point in "data". The default is None.

    Returns
    -------
    float, float
        Average and standard deviation of "data".

    """
    avg = np.average(data, weights=weights)
    var = np.average((data - avg) ** 2, weights=weights)
    return avg, math.sqrt(var)


def is_contained(bbox, voxel):
    """
    Checks whether "voxel" is contained within "bbox".

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxel : ArrayLike
        Voxel coordinate to be checked.

    Returns
    -------
    bool
        Inidcation of whether "voxel" is contained in "bbox".

    """
    above = any(voxel >= bbox["max"])
    below = any(voxel < bbox["min"])
    return False if above or below else True


def is_list_contained(bbox, voxels):
    """
    Checks whether every element in "xyz_list" is contained in "bbox".

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxels
        List of xyz coordinates to be checked.

    Returns
    -------
    bool
        Indication of whether every element in "voxels" is contained in
        "bbox".

    """
    return all([is_contained(bbox, voxel) for voxel in voxels])


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


# --- dictionary utils ---
def remove_items(my_dict, keys):
    """
    Removes dictionary items corresponding to "keys".

    Parameters
    ----------
    my_dict : dict
        Dictionary to be edited.
    keys : list
        List of keys to be deleted from "my_dict".

    Returns
    -------
    dict
        Updated dictionary.

    """
    for key in keys:
        if key in my_dict:
            del my_dict[key]
    return my_dict


def find_best(my_dict, maximize=True):
    """
    Given a dictionary where each value is either a list or int (i.e. cnt),
    finds the key associated with the longest list or largest integer.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be searched.
    maximize : bool, optional
        Indication of whether to find the largest value or highest vote cnt.

    Returns
    -------
    hashable data type
        Key associated with the longest list or largest integer in "my_dict".

    """
    best_key = None
    best_vote_cnt = 0 if maximize else np.inf
    for key in my_dict.keys():
        val_type = type(my_dict[key])
        vote_cnt = my_dict[key] if val_type == float else len(my_dict[key])
        if vote_cnt > best_vote_cnt and maximize:
            best_key = key
            best_vote_cnt = vote_cnt
        elif vote_cnt < best_vote_cnt and not maximize:
            best_key = key
            best_vote_cnt = vote_cnt
    return best_key


# --- miscellaneous ---
def count_fragments(fragments_pointer, min_size=0):
    """
    Counts the number of fragments in a given predicted segmentation.

    Parameters
    ----------
    swc_pointer : dict, list, str
        Object that points to swcs to be read, see class documentation for
        details.
    min_size : float, optional

    """
    # Extract fragments
    graph_loader = gutil.GraphLoader(min_size=min_size, progress_bar=True)
    graph = graph_loader.run(fragments_pointer)

    # Report results
    print("# Connected Components:", nx.number_connected_components(graph))
    print("# Nodes:", graph.number_of_nodes())
    print("# Edges:", graph.number_of_edges())
    del graph


def time_writer(t, unit="seconds"):
    """
    Converts a runtime "t" to a larger unit of time if applicable.

    Parameters
    ----------
    t : float
        Runtime.
    unit : str, optional
        Unit of time that "t" is expressed in.

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


def get_swc_id(path):
    """
    Gets segment id of the swc file at "path".

    Parameters
    ----------
    path : str
        Path to swc file.

    Returns
    -------
    str
        Segment id.

    """
    filename = path.split("/")[-1]
    return filename.split(".")[0]


def numpy_to_hashable(arr):
    """
    Converts a numpy array to a hashable data structure.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    list
        Hashable items from "arr".

    """
    return [tuple(item) for item in arr.tolist()]


def reformat_number(number):
    """
    Reformats large number to have commas.

    Parameters
    ----------
    number : float
        Number to be reformatted.

    Returns
    -------
    str
        Reformatted number.

    """
    return f"{number:,}"


def get_memory_usage():
    """
    Gets the current memory usage in gigabytes.

    Parameters
    ----------
    None

    Returns
    -------
    float
        Current memory usage in gigabytes.

    """
    return psutil.virtual_memory().used / 1e9


def get_memory_available():
    """
    Gets the available memory in gigabytes.

    Parameters
    ----------
    None

    Returns
    -------
    float
        Available memory usage in gigabytes.

    """
    return psutil.virtual_memory().available / 1e9


def spaced_idxs(container, k):
    """
    Generates an array of indices based on a specified step size and ensures
    the last index is included.

    Parameters:
    ----------
    container : iterable
        An iterable (e.g., list, array) from which the length is determined.
        The length  of this container dictates the range of generated indices.
    k : int
        Step size for generating indices.

    Returns:
    -------
    numpy.ndarray
        Array of indices starting from 0 up to (but not including) the length
        of "container" spaced by "k". The last index before the length of
        "container" is guaranteed to be included in the output.

    """
    idxs = np.arange(0, len(container) + k, k)[:-1]
    if len(container) % 2 == 0:
        idxs = np.append(idxs, len(container) - 1)
    return idxs