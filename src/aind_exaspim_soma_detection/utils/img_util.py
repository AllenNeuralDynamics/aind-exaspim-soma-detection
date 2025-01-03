"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

import matplotlib.pyplot as plt
import numpy as np
import s3fs
import zarr

ANISOTROPY = [0.748, 0.748, 1.0]


def open_img(s3_prefix):
    """
    Opens an image stored in an S3 bucket as a Zarr array.

    Parameters:
    -----------
    s3_prefix : str
        The prefix (or path) within the S3 bucket where the image is stored.

    Returns:
    --------
    zarr.core.Array
        A Zarr object representing an image.

    """
    store = s3fs.S3Map(root=s3_prefix, s3=s3fs.S3FileSystem())
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
        Dimensions of the patch to extract.
    from_center : bool, optional
        Indicates whether the given voxel is the center or top, left, front
        corner of the patch to be extracted.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.

    """
    start, end = get_start_end(voxel, shape, from_center=from_center)
    return img[0, 0, start[2]: end[2], start[1]: end[1], start[0]: end[0]]


def sliding_window_coords_3d(img, window_size, overlap):
    # Calculate stride based on the overlap and window size
    stride = tuple(w - o for w, o in zip(window_size, overlap))
    z_stride, y_stride, x_stride = stride

    # Get dimensions of the window
    _, _, z_dim, y_dim, x_dim = img.shape
    z_win, y_win, x_win = window_size

    # Loop over the  with the sliding window
    coords = []
    for x in range(0, x_dim - x_win + 1, x_stride):
        for y in range(0, y_dim - y_win + 1, y_stride):
            for z in range(0, z_dim - z_win + 1, z_stride):
                coords.append((x, y, z))
    return coords


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the image patch to be read.

    Parameters
    ----------
    voxel : tuple
        Voxel coordinate that specifies either the center or top, left, front
        corner of the patch to be read.
    shape : tuple
        Shape (dimensions) of the patch to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the patch or the starting point. The default is True.

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


# --- coordinate conversions ---
def to_physical(voxel):
    """
    Converts the given coordinate from voxels to physical space.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinate to be converted.

    Returns
    -------
    Tuple[int]
        Physical coordinates of "voxel".

    """
    return tuple([voxel[i] * ANISOTROPY[i] for i in range(3)])


def to_voxels(xyz, multiscale=0):
    """
    Converts the given coordinate from physical to voxel space.

    Parameters
    ----------
    xyz : numpy.ndarray
        xyz point to be converted to voxel coordinates.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        The default is 0.

    Returns
    -------
    numpy.ndarray
        Coordinate converted to voxels.

    """
    multiscale = 1.0 / 2**multiscale
    voxel = multiscale * (xyz / np.array(ANISOTROPY))
    return np.round(voxel).astype(int)


def local_to_physical(local_voxel, offset, multiscale=0):
    global_voxel = np.array([v + o for v, o in zip(local_voxel, offset)])
    return to_physical(global_voxel * 2**multiscale)


# --- visualizations ---
def plot_mips(img, prefix="", clip_bool=False):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        axs[i].imshow(get_mip(img, axis=i, clip_bool=clip_bool))
        axs[i].set_title(prefix + axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def rescale(img, clip_bool=True):
    if clip_bool:
        img = np.clip(img, 0, np.percentile(img, 99))
    img -= np.min(img)
    img = (2**16 - 1) * (img / np.max(img))
    return (img).astype(np.uint16)


def get_mip(img, axis=0, clip_bool=False):
    mip = np.max(img, axis=axis)
    mip = rescale(mip, clip_bool=clip_bool)
    return mip


def mark_voxel(img, voxel, value=1):
    img[voxel] = value
    return img
