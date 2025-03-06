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


# --- Visualizations ---
def plot_mips(img, clip_bool=False):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.
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
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def get_mip(img_patch, axis=0, clip_bool=False):
    """
    Computes the Maximum Intensity Projection (MIP) along a specified axis and
    rescales the resulting image.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image to generate MIPs from.
    axis : int, optional
        The axis along which to compute the MIP. The default is 0.
    clip_bool : bool, optional
        If True, the resulting MIP will be clipped to the range [0, 1] during
        rescaling. If False, no clipping is applied. The default is False.

    Returns
    -------
    numpy.ndarray
        Maximum Intensity Projection (MIP) along the specified axis, after
        rescaling.

    """
    mip = np.max(img_patch, axis=axis)
    return rescale(mip, clip_bool=clip_bool)


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


def rescale(img_patch, clip_bool=True):
    """
    Rescales the input image to a [0, 2**16 - 1] intensity range, with optional
    clipping at the 99th percentile.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image patch to be rescaled.
    clip_bool : bool, optional
        If True, the resulting MIP will be clipped at the 99th percentile. The
        default is True.

    Returns
    -------
    numpy.ndarray
        Rescaled image.

    """
    # Clip image
    if clip_bool:
        img_patch = np.clip(img_patch, 0, np.percentile(img_patch, 99))

    # Rescale image
    img_patch = (2**16 - 1) * normalize(img_patch)
    return img_patch.astype(np.uint16)
