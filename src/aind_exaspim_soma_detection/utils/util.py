"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import as_completed, ThreadPoolExecutor

from io import StringIO
from random import sample
from zipfile import ZipFile

import ast
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


def read_soma_locations(path):
    xyz_list = list()
    for xyz_str in read_txt(path):
        xyz_list.append(ast.literal_eval(xyz_str))
    return xyz_list


def write_json(path, my_dict):
    """
    Writes the contents in the given dictionary to a json file at "path".

    Parameters
    ----------
    path : str
        Path where JSON file is stored.
    my_dict : dict
        Dictionary to be written to a JSON.

    Returns
    -------
    None

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
        The list of items to write to the file.

    Returns
    -------
    None

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
        Color of nodes. The default is None.
    radius : float, optional
        Radius of point. The default is 5um.

    Returns
    -------
    None

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


def read_s3_txt_file(bucket_name, file_path):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.get_object(Bucket=bucket_name, Key=file_path)
    return response['Body'].read().decode('utf-8').splitlines()


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


# --- S3 Soma utils ---
def load_somas_locations(brain_id, filtered=False):
    bucket_name = 'aind-msma-morphology-data'
    file_path = find_soma_result_prefix(brain_id, filtered=filtered)
    lines = read_s3_txt_file(bucket_name, file_path)
    return [ast.literal_eval(xyz) for xyz in lines]


def find_soma_result_prefix(brain_id, filtered=False):
    # Find soma results for brain_id
    bucket_name = 'aind-msma-morphology-data'
    soma_prefix = f"exaspim_soma_detection/{brain_id}"
    prefix_list = list_s3_prefixes(bucket_name, soma_prefix)

    # Find most recent result
    if prefix_list:
        dirname = find_most_recent_dirname(prefix_list)
        pre = "filtered-" if filtered else ""
        filename = f"{pre}somas-{brain_id}.txt"
        return os.path.join(soma_prefix, dirname, filename)
    else:
        return None

    
def find_most_recent_dirname(results_prefix_list):
    dates = list()
    for prefix in results_prefix_list:
        dirname = prefix.split("/")[-2]
        dates.append(dirname.replace("results_", ""))
    return "results_" + sorted(dates)[-1]


# --- Miscellaneous ---
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
