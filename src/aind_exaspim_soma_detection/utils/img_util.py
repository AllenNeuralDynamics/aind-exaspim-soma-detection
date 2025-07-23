"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np
import os
import s3fs
import zarr

from aind_exaspim_soma_detection.utils import util

ANISOTROPY = [0.748, 0.748, 1.0]


def open_img(path):
    """
    Opens an image stored in an S3 bucket as a Zarr array.

    Parameters
    ----------
    path : str
        Path to image stored in an S3 bucket.

    Returns
    -------
    zarr.core.Array
        Zarr object representing an image.
    """
    store = s3fs.S3Map(root=path, s3=s3fs.S3FileSystem())
    return zarr.open(store, mode="r")


def get_patch(img, voxel, shape, from_center=True):
    """
    Extracts a patch from an image based on the given voxel coordinate and
    patch shape.

    Parameters
    ----------
    img : zarr.core.Array
         Zarr object representing an image.
    voxel : Tuple[int]
        Center of patch to be extracted
    shape : Tuple[int]
        Shape of patch to be extracted.
    from_center : bool, optional
        Indicates whether the given voxel is the center or top, left, front
        corner of the patch to be extracted. Default is True.

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
        Overlap between adjacent windows.

    Returns
    -------
    List[Tuple[int]]
        Voxel coordinates representing the front-top-left corner.
    """
    # Calculate stride based on the overlap and window size
    stride = tuple(w - o for w, o in zip(window_shape, overlap))
    i_stride, j_stride, k_stride = stride

    # Get dimensions of the window
    _, _, i_dim, j_dim, k_dim = img.shape
    i_win, j_win, k_win = window_shape

    # Loop over the  with the sliding window
    voxels = []
    for i in range(0, i_dim - i_win + 1, i_stride):
        for j in range(0, j_dim - j_win + 1, j_stride):
            for k in range(0, k_dim - k_win + 1, k_stride):
                voxels.append((i, j, k))
    return voxels


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
        Physical coordinate of the given voxel.
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
        Voxel coordinate of the given physical coordinate.
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
    """
    Finds the image prefix corresponding to the given brain ID.

    Parameters
    ----------
    brain_id : str
        Brain ID used to find image prefix.

    Returns
    -------
    str
        Image prefix corresponding to the given brain ID.
    """
    # Initializations
    bucket_name = "aind-open-data"
    prefixes = util.list_s3_bucket_prefixes(
        "aind-open-data", keyword="exaspim"
    )

    # Get possible prefixes
    valid_prefixes = list()
    for prefix in prefixes:
        # Check for new naming convention
        if util.exists_in_prefix(bucket_name, prefix, "fusion"):
            prefix = os.path.join(prefix, "fusion")
        
        # Check if prefix is valid
        if is_valid_prefix(bucket_name, prefix, brain_id):
            valid_prefixes.append(
                os.path.join("s3://aind-open-data", prefix, "fused.zarr")
            )
    return find_functional_img_prefix(valid_prefixes)


def is_valid_prefix(bucket_name, prefix, brain_id):
    # Quick checks
    is_test = "test" in prefix.lower()
    has_correct_id = str(brain_id) in prefix
    if not has_correct_id or is_test:
        return False

    # Check inside prefix - old convention
    if util.exists_in_prefix(bucket_name, prefix, "fused.zarr"):
        img_prefix = os.path.join(prefix, "fused.zarr")
        multiscales = util.list_s3_prefixes(bucket_name, img_prefix)
        multiscales = [s.split("/")[-2] for s in multiscales]
        for s in map(str, range(0, 8)):
            if s not in multiscales:
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
    Generates a binary image where the given voxels are marked.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the output image.
    voxels : List[Tuple[int]]
        Voxel coordinates to be marked as detected in the output image.

    Returns
    -------
    numpy.ndarray
        Binary image where the given voxels are marked.
    """
    detections_img = np.zeros(shape)
    for voxel in voxels:
        voxel = tuple([int(v) for v in voxel])
        detections_img[voxel] = 1
    return detections_img


# --- Fit gaussian to image ---
def fit_gaussian_3d(img_patch, std=2):
    """
    Fits a 3D Gaussian to an image patch.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D image that Gaussian is to be fitted to.
    std : float, optional
        Estimate of standard devation of Gaussian to be fit. Default is 2.

    Returns
    -------
    tuple
        Parameters of the fitted Gaussian and voxel coordinates.
    """
    center = [s // 2 for s in img_patch.shape]
    initial_guess = (
        center[0], center[1], center[2],
        std, std, std,
        np.max(img_patch), np.min(img_patch)
    )
    return fit(img_patch, gaussian_3d, initial_guess)


def fit_rotated_gaussian_3d(img_patch):
    center = [s // 2 for s in img_patch.shape]
    initial_guess = (
        center[0], center[1], center[2],
        1e-2, 0, 0,
        1e-2, 0,
        1e-2,
        np.max(img_patch), np.min(img_patch)
    )
    return fit(img_patch, rotated_gaussian_3d, initial_guess)


def fit(img_patch, my_func, initial_guess):
    """
    Fits a function (e.g. gaussian) to an image.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D array representing an image.
    my_func : callable
        Function to be fit to image.
    initial_guess : numpy.ndarray
        Initial guess of parameters.

    Returns
    -------
    params : numpy.ndarray
        Parameters of fitted function
    voxels : numpy.ndarray
        Flattened arrays of voxel coordinates.
    """
    try:
        voxels = generate_img_coords(img_patch.shape)
        img_vals = img_patch.ravel()
        params, _ = curve_fit(my_func, voxels, img_vals, p0=initial_guess)
    except RuntimeError:
        params = np.zeros(len(initial_guess))
    return params, voxels


def compute_fit_score(img_patch, params, voxels):
    """
    Evaluates the quality of a fitted function by computing the correlation
    coefficient between the image and fitted values.

    Parameters
    ----------
    img_patch : numpy.ndarray
        A 3D array representing an image.
    params : numpy.ndarray
        Parameters of the fitted Gaussian.
    voxels : Tuple[numpy.ndarray]
        Flattened arrays of voxel coordinates.

    Returns
    -------
    float
        Correlation coefficient between the image and fitted values.
    """
    gaussian = gaussian_3d if len(params) == 8 else rotated_gaussian_3d
    fitted = gaussian(voxels, *params).reshape(img_patch.shape).flatten()
    actual = img_patch.flatten()
    return np.corrcoef(actual, fitted)[0, 1]


def gaussian_3d(
    coords, x0, y0, z0, sigma_x, sigma_y, sigma_z, amplitude, offset
):
    """
    Computes the values of a 3D Gaussian at the given coordinates.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates that Gaussian is to be evaluated.
    x0, y0, z0 : float
        Mean of Gaussian.
    sigma_x, sigma_y, sigma_z : float
        Standard deviations of Gaussian.
    amplitude : float
        Peak value (amplitude) of Gaussian at the center.
    offset : float
        Constant value added to Gaussian that represents the baseline offset.

    Returns
    -------
    numpy.ndarray
        Computed values of the 3D Gaussian at the given coordinates. Note that
        these values are flattened from a 3D grid to a 1D array.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    value = offset + amplitude * np.exp(
        -(
            ((x - x0) ** 2) / (2 * sigma_x**2)
            + ((y - y0) ** 2) / (2 * sigma_y**2)
            + ((z - z0) ** 2) / (2 * sigma_z**2)
        )
    )
    return value.ravel()


def rotated_gaussian_3d(
    coords, x0, y0, z0, a11, a12, a13, a22, a23, a33, A, B
):
    # Refactor coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Construct quadratic form
    quad = (
        a11*dx**2 + 2*a12*dx*dy + 2*a13*dx*dz +
        a22*dy**2 + 2*a23*dy*dz + a33*dz**2
    )
    return A * np.exp(-0.5 * quad) + B


def rotated_gaussian_3d_mask(
    shape, voxels, x0, y0, z0, a11, a12, a13, a22, a23, a33, threshold=4.0
):
    """
    Computes a binary mask of voxels within a specified Mahalanobis distance
    (default: 2 standard deviations => threshold=4) from the Gaussian center.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of image that the given coordinates coorespond to.
    voxels : numpy.ndarray
        Voxel coordinates.
    x0, y0, z0 : float
        Center of the Gaussian.
    a11, a12, a13, a22, a23, a33 : float
        Elements of the symmetric positive-definite matrix defining the
        quadratic form. This matrix is the inverse of the covariance matrix.
    threshold : float
        Mahalanobis distance squared (e.g., 4.0 for 2 standard deviations).

    Returns
    -------
    mask : ndarray of shape (N,)
        Boolean array where True indicates the voxel is within the threshold.
    """

    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    dx = x - x0
    dy = y - y0
    dz = z - z0
    quad = (
        a11*dx**2 + 2*a12*dx*dy + 2*a13*dx*dz +
        a22*dy**2 + 2*a23*dy*dz + a33*dz**2
    )
    return (quad <= threshold).reshape(shape)


# --- Utils ---
def generate_img_coords(shape):
    """
    Generates all voxel coordinates of an image patch given its shape.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of image patch to generate voxel coordinates for.

    Returns
    -------
    numpy.ndarray
        Array containing all voxel coordinates of an image patch.
    """
    grid = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    return np.stack(grid, axis=-1).reshape(-1, 3)


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
