"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from random import sample

import boto3
import json
import os
import shutil


# --- OS utils ---
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


def write_to_list(path, my_list):
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
    Writes an SWC file.

    Parameters
    ----------
    path : str
        Path on local machine that SWC file will be written to.
    xyz : ArrayLike
        xyz coordinate to be written to an SWC file.
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
def exists_in_prefix(bucket_name, prefix, name):
    """
    Checks if a given filename is in a prefix.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket to search.
    prefix : str
        S3 prefix to search within.
    name : str
        Filename to search for.

    Returns
    -------
    bool
        Indiciation of whether a given file is in a prefix.

    """
    prefixes = list_s3_prefixes(bucket_name, prefix)
    return sum([1 for prefix in prefixes if name in prefix]) > 0


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

    Returns
    -------
    None

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

    Returns
    -------
    None

    """
    s3 = boto3.client("s3")
    s3.upload_file(source_path, bucket_name, destination_path)


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
