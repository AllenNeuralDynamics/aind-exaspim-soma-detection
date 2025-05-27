"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import s3fs
import zarr

from aind_exaspim_soma_detection.utils import util

ANISOTROPY = [0.748, 0.748, 1.0]


def open_img(prefix):
    """
    Opens an image stored in an S3 bucket as a Zarr array.

    Parameters
    ----------
    prefix : str
        Prefix (or path) within the S3 bucket where the image is stored.

    Returns
    -------
    zarr.core.Array
        A Zarr object representing an image.

    """
    store = s3fs.S3Map(root=prefix, s3=s3fs.S3FileSystem())
    return zarr.open(store, mode="r")


def get_patch(img, voxel, shape, from_center=True):
    """
    Extracts a patch from an image based on the given voxel coordinate and
    patch shape.

    Parameters
    ----------
    img : zarr.core.Array
         A Zarr object representing an image.
    voxel : Tuple[int]
        Voxel coordinate used to extract patch.
    shape : Tuple[int]
        Shape of the image patch to extract.
    from_center : bool, optional
        Indicates whether the given voxel is the center or top, left, front
        corner of the patch to be extracted.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.

    """
    # Get image patch coordiantes
    start, end = get_start_end(voxel, shape, from_center=from_center)
    valid_start = any([s >= 0 for s in start])
    valid_end = any([e < img.shape[i + 2] for i, e in enumerate(end)])

    # Get image patch
    if valid_start and valid_end:
        return img[
            0, 0, start[0]: end[0], start[1]: end[1], start[2]: end[2]
        ]
    else:
        return np.ones(shape)


def calculate_offsets(img, window_shape, overlap):
    """
    Generates a list of 3D coordinates representing the front-top-left corner
    by sliding a window over a 3D image, given a specified window size and
    overlap between adjacent windows.

    Parameters
    ----------
    img : zarr.core.Array
        Input 3D image.
    window_shape : Tuple[int]
        Shape of the sliding window.
    overlap : Tuple[int]
        Overlap between adjacent sliding windows.

    Returns
    -------
    List[Tuple[int]]
        List of 3D voxel coordinates that represent the front-top-left corner.

    """
    # Calculate stride based on the overlap and window size
    stride = tuple(w - o for w, o in zip(window_shape, overlap))
    i_stride, j_stride, k_stride = stride

    # Get dimensions of the window
    _, _, i_dim, j_dim, k_dim = img.shape
    i_win, j_win, k_win = window_shape

    # Loop over the  with the sliding window
    coords = []
    for i in range(0, i_dim - i_win + 1, i_stride):
        for j in range(0, j_dim - j_win + 1, j_stride):
            for k in range(0, k_dim - k_win + 1, k_stride):
                coords.append((i, j, k))
    return coords


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the image patch to be read.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate that specifies either the center or front-top-left
        corner of the patch to be read.
    shape : Tuple[int]
        Shape of the image patch to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the patch or the front-top-left corner. The default is True.

    Return
    ------
    Tuple[List[int]]
        Start and end indices of the image patch to be read.

    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# --- Coordinate Conversions ---
def to_physical(voxel, multiscale):
    """
    Converts the given coordinate from voxels to physical space.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    multiscale
        Level in the image pyramid that the voxel coordinate must index into.


    Returns
    -------
    Tuple[int]
        Physical coordinate of "voxel".

    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * ANISOTROPY[i] * 2**multiscale for i in range(3)])


def to_voxels(xyz, multiscale):
    """
    Converts the given coordinate from physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordiante to be converted to a voxel coordinate.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.

    Returns
    -------
    numpy.ndarray
        Voxel coordinate of the input.

    """
    scaling_factor = 1.0 / 2**multiscale
    voxel = scaling_factor * (xyz / np.array(ANISOTROPY))
    return np.round(voxel).astype(int)[::-1]


def local_to_physical(local_voxel, offset, multiscale):
    """
    Converts a local voxel coordinate to a physical coordinate in global
    space.

    Parameters
    ----------
    local_voxel : Tuple[int]
        Local voxel coordinate in an image patch.
    offset : Tuple[int]
        Offset from the local coordinate system to the global coordinate
        system.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.

    Returns
    -------
    numpy.ndarray
        Physical coordinate.

    """
    global_voxel = np.array([v + o for v, o in zip(local_voxel, offset)])
    return to_physical(global_voxel, multiscale)


# --- Image Prefix Search ---
def get_img_prefix(brain_id, img_prefix_path=None):
    # Check prefix path
    if img_prefix_path:
        prefix_lookup = util.read_json(img_prefix_path)
        if brain_id in prefix_lookup:
            return prefix_lookup[brain_id]

    # Search for prefix path
    result = find_img_prefix(brain_id)
    if len(result) == 1:
        prefix = result[0] + "/"
        if img_prefix_path:
            prefix_lookup[brain_id] = prefix
            util.write_json(img_prefix_path, prefix_lookup)
        return prefix

    raise Exception(f"Image Prefixes Found - {result}")


def find_img_prefix(brain_id):
    # Get possible prefixes
    bucket_name = "aind-open-data"
    prefixes = util.list_s3_bucket_prefixes(bucket_name, keyword="exaspim")
    valid_prefixes = list()
    for prefix in prefixes:
        if is_valid_prefix(bucket_name, prefix, brain_id):
            valid_prefixes.append(
                os.path.join(f"s3://{bucket_name}", prefix, "fused.zarr")
            )
    return find_functional_img_prefix(valid_prefixes)


def is_valid_prefix(bucket_name, prefix, brain_id):
    # Quick checks
    is_test = "test" in prefix.lower()
    has_correct_id = str(brain_id) in prefix
    if not has_correct_id or is_test:
        return False

    # Check inside prefix
    if util.exists_in_prefix(bucket_name, prefix, "fused.zarr"):
        img_prefix = os.path.join(prefix, "fused.zarr")
        subprefixes = util.list_s3_prefixes(bucket_name, img_prefix)
        subprefixes = [p.split("/")[-2] for p in subprefixes]
        for i in range(0, 7):
            if str(i) not in subprefixes:
                return False
    return True


def find_functional_img_prefix(prefixes):
    # Filter img prefixes that fail to open
    functional_prefixes = list()
    for prefix in prefixes:
        try:
            root = os.path.join(prefix, str(0))
            store = s3fs.S3Map(root=root, s3=s3fs.S3FileSystem(anon=True))
            img = zarr.open(store, mode="r")
            if np.max(img.shape) > 25000:
                functional_prefixes.append(prefix)
        except:
            pass
    return functional_prefixes


# --- Visualizations ---
def plot_mips(img, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.

    Returns
    -------
    None

    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        mip = np.max(img, axis=i)
        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def get_detections_img(shape, voxels):
    """
    Converts a list of voxel coordinates into a binary detection image, where
    detected voxels are marked.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the output detection image.
    voxels : List[Tuple[int]]
        List of voxel coordinates to be marked as detected in the output
        image.

    Returns
    -------
    numpy.ndarray
        A binary detection image, where each voxel in "voxels" is marked with
        1 and all other positions are set to 0.

    """
    detections_img = np.zeros(shape)
    for voxel in voxels:
        voxel = tuple([int(v) for v in voxel])
        detections_img[voxel] = 1
    return detections_img


# --- Utils ---
def get_nbs(voxel, shape):
    """
    Gets the neighbors of a given voxel in a 3D grid with respect to
    26-connectivity.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate for which neighbors are to be found.
    shape : Tuple[int]
        Shape of the 3D grid. This is used to ensure that neighbors are
        within the grid boundaries.

    Returns
    -------
    List[Tuple[int]]
        Voxel coordinates of the neighboring voxels.

    """
    x, y, z = voxel
    nbs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Skip the given voxel
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                # Add neighbor
                nb = (x + dx, y + dy, z + dz)
                if is_inbounds(nb, shape):
                    nbs.append(nb)
    return nbs


def is_inbounds(voxel, shape):
    """
    Checks if a given voxel is within the bounds of a 3D grid.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate to be checked.
    shape : Tuple[int]
        Shape of the 3D grid.

    Returns
    -------
    bool
        Indication of whether the given voxel is within the bounds of the
        grid.

    """
    x, y, z = voxel
    height, width, depth = shape
    if 0 <= x < height and 0 <= y < width and 0 <= z < depth:
        return True
    else:
        return False


def normalize(img_patch):
    """
    Rescales the input image to a [0, 1] intensity range.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image patch to be normalized.

    Returns
    -------
    numpy.ndarray
        Normalized image.

    """
    img_patch -= np.min(img_patch)
    return img_patch / np.max(img_patch)
