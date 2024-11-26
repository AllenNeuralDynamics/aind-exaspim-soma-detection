"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

import matplotlib.pyplot as plt
import numpy as np

ANISOTROPY = [0.748, 0.748, 1.0]


def sliding_window_coords_3d(img, window_size, overlap):
    # Calculate the stride based on the overlap and window size
    stride = tuple(w - o for w, o in zip(window_size, overlap))

    # Get dimensions of the  and window
    _, _, z_dim, y_dim, x_dim = img.shape
    z_win, y_win, x_win = window_size
    z_stride, y_stride, x_stride = stride

    # Loop over the  with the sliding window
    coords = []
    for x in range(0, x_dim - x_win + 1, x_stride):
        for y in range(0, y_dim - y_win + 1, y_stride):
            for z in range(0, z_dim - z_win + 1, z_stride):
                coords.append((x, y, z))
    return coords


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    voxel : tuple
        Voxel coordinate that specifies either the center or top, left, front
        corner of the chunk to be read.
    shape : tuple
        Shape (dimensions) of the chunk to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the chunk or the starting point. The default is True.

    Return
    ------
    tuple[list[int]]
        Start and end indices of the chunk to be read.

    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# --- coordinate conversions ---
def to_world(voxel, shift=[0, 0, 0]):
    """
    Converts coordinates from voxels to world.

    Parameters
    ----------
    coord : numpy.ndarray
        Coordinate to be converted.
    shift : list, optional
        Shift to be applied to "coord". The default is [0, 0, 0].

    Returns
    -------
    tuple
        Converted coordinates.

    """
    return tuple([voxel[i] * ANISOTROPY[i] - shift[i] for i in range(3)])


def to_voxels(xyz, downsample_factor=0):
    """
    Converts coordinates from world to voxel.

    Parameters
    ----------
    xyz : numpy.ndarray
        xyz coordinate to be converted to voxels.
    anisotropy : list, optional
        Anisotropy to be applied to values of interest. The default is
        [1.0, 1.0, 1.0].
    downsample_factor : int, optional
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into. The default is 0.

    Returns
    -------
    numpy.ndarray
        Coordinates converted to voxels.

    """
    downsample_factor = 1.0 / 2 ** downsample_factor
    voxel = downsample_factor * (xyz / np.array(ANISOTROPY))
    return np.round(voxel).astype(int)


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
