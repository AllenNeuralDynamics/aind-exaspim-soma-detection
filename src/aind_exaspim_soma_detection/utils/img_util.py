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


def open_img(prefix):
    """
    Opens an image stored in an S3 bucket as a Zarr array.

    Parameters:
    -----------
    prefix : str
        Prefix (or path) within the S3 bucket where the image is stored.

    Returns:
    --------
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


def sliding_window_coords_3d(img, window_shape, overlap):
    """
    Generates a list of 3D coordinates representing the front-top-left corner
    of each sliding window over a 3D image, given a specified window size and
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
    z_stride, y_stride, x_stride = stride

    # Get dimensions of the window
    _, _, z_dim, y_dim, x_dim = img.shape
    z_win, y_win, x_win = window_shape

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
        Voxel coordinate that specifies either the center or front-top-left
        corner of the patch to be read.
    shape : tuple
        Shape (dimensions) of the patch to be read.
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
    scaling_factor = 1.0 / 2 ** multiscale
    voxel = scaling_factor * (xyz / np.array(ANISOTROPY))
    return np.round(voxel).astype(int)


def local_to_physical(local_voxel, offset, multiscale=0):
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
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        The default is 0.

    Returns
    -------
    numpy.ndarray
        Physical coordinate.

    """
    global_voxel = np.array([v + o for v, o in zip(local_voxel, offset)])
    return to_physical(global_voxel * 2**multiscale)


# --- Visualizations ---
def plot_mips(img, prefix="", clip_bool=False):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.
    prefix : str, optional
        String to be added as a prefix to the titles of the MIP plots. The
        default is an empty string.
    clip_bool : bool, optional
        If True, the resulting MIP will be clipped to the range [0, 1] during
        rescaling. If False, no clipping is applied. The default is False.

    Returns
    -------
    None

    """
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
    """
    Rescales the input image to a [0, 65535] intensity range, with optional
    clipping of extreme values.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.
    clip_bool : bool, optional
        If True, the resulting MIP will be clipped to the range [0, 1] during
        rescaling. If False, no clipping is applied. The default is False.

    Returns
    -------
    numpy.ndarray
        Rescaled image.

    """
    # Clip image
    if clip_bool:
        img = np.clip(img, 0, np.percentile(img, 99))

    # Rescale image
    img -= np.min(img)
    img = (2**16 - 1) * (img / np.max(img))
    return img.astype(np.uint16)


def get_mip(img, axis=0, clip_bool=False):
    """
    Computes the Maximum Intensity Projection (MIP) along a specified axis and
    rescales the resulting image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image to generate MIPs from.
    axis : int, optional
        The axis along which to compute the maximum intensity projection. The
        default is 0.
    clip_bool : bool, optional
        If True, the resulting MIP will be clipped to the range [0, 1] during
        rescaling. If False, no clipping is applied. The default is False.

    Returns
    -------
    numpy.ndarray
        Maximum Intensity Projection (MIP) along the specified axis, after
        rescaling.

    """
    mip = np.max(img, axis=axis)
    mip = rescale(mip, clip_bool=clip_bool)
    return mip


def get_detections_img(shape, voxels):
    """
    Converts a list of voxel coordinates into a binary detection image, where
    detected voxels are marked.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the output detection image.
    voxels : List[Tuple[int]
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
