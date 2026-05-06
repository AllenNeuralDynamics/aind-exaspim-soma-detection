"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from scipy.optimize import curve_fit
import tensorstore as ts

import matplotlib.pyplot as plt
import numpy as np

from aind_exaspim_soma_detection.utils import util


class TensorStoreImage:
    """
    Class that uses the TensorStore library for image IO operations.
    """

    def __init__(self, img_path):
        """
        Instantiates a TensorStoreImage object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        # Open image
        self.img = ts.open(self.get_spec(img_path)).result()
        self.img_path = img_path

        # Check dimensions
        while self.img.ndim < 5:
            self.img = self.img[ts.newaxis, ...]

    # --- Core Routines ---
    def read(self, voxel, shape, is_center=True):
        """
        Reads the image patch specified by the given slices.

        Parameters
        ----------
        voxel : Tuple[int]
            Voxel coordinate used as reference to extract patch.
        shape : Tuple[int]
            Shape of patch to be extracted.
        is_center : bool, optional
            Indicates whether the given voxel is the center or top-left-front
            corner of the patch to be extracted. Default is True.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        try:
            _get_slices = get_center_slices if is_center else get_slices
            slices = _get_slices(voxel, shape)
            patch = self.img[slices].read().result()
        except ValueError:
            print(f"Error reading {slices} from img w/ shape {self.shape()}")
            patch = np.zeros(tuple(s.stop - s.start for s in slices))
        return patch

    # --- Helpers ---
    def generate_offsets(self, window_shape, overlap):
        """
        Generates a list of 3D coordinates representing the front-top-left corner
        by sliding a window over a 3D image, given a specified window size and
        overlap between adjacent windows.

        Parameters
        ----------
        window_shape : Tuple[int]
            Shape of the sliding window.
        overlap : Tuple[int]
            Overlap between adjacent windows.

        Returns
        -------
        Iterator[Tuple[int]]
            Voxel coordinates representing the front-top-left corner.
        """
        # Calculate stride based on the overlap and window size
        stride = tuple(w - o for w, o in zip(window_shape, overlap))
        i_stride, j_stride, k_stride = stride

        # Get dimensions of the window
        _, _, i_dim, j_dim, k_dim = self.shape()
        i_win, j_win, k_win = window_shape

        # Loop over img with the sliding window
        for i in range(0, i_dim - i_win + 1, i_stride):
            for j in range(0, j_dim - j_win + 1, j_stride):
                for k in range(0, k_dim - k_win + 1, k_stride):
                    yield (i, j, k)

    def get_spec(self, img_path):
        """
        Creates a TensorStore specification for opening the image at the
        given path.

        Parameters
        ----------
        img_path : str
            Path to image to be opened.

        Returns
        -------
        spec : dict
            TensorStore specification for opening the image at the given path.
        """
        bucket_name, relative_path = util.parse_cloud_path(img_path)
        spec = {
            "driver": get_driver(img_path),
            "kvstore": {
                "driver": get_storage_driver(img_path),
                "bucket": bucket_name,
                "path": relative_path,
            },
            "context": {
                "cache_pool": {"total_bytes_limit": 1000000000},
                "cache_pool#remote": {"total_bytes_limit": 1000000000},
                "data_copy_concurrency": {"limit": 8},
            },
        }
        return spec

    def shape(self):
        """
        Gets the shape of the image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


# --- Coordinate Conversions ---
def to_physical(voxel, multiscale, anisotropy=(0.748, 0.748, 1.0)):
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
    return tuple([voxel[i] * anisotropy[i] * 2**multiscale for i in range(3)])


def to_voxels(xyz, multiscale, anisotropy=(0.748, 0.748, 1.0)):
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
    voxel = scaling_factor * (xyz / np.array(anisotropy))
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
    voxel = np.array([v + o for v, o in zip(local_voxel, offset)])
    return to_physical(voxel, multiscale)


# --- Visualizations ---
def plot_mips(img, output_path=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.
    output_path : None or str, optional
        Path to save MIPs as a PNG if provided. Default is None.
    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        mip = np.max(img, axis=i)
        axs[i].imshow(mip)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)


def plot_slices(img, output_path=None, vmax=None):
    """
    Plots the middle slice of a 3D image along the XY, XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Image to generate MIPs from.
    output_path : None or str, optional
        Path that plot is saved to if provided. Default is None.
    vmax : None or float, optional
        Brightness intensity used as upper limit of the colormap. Default is
        None.
    """
    # Get middle slice
    shape = img.shape[2:] if len(img.shape) == 5 else img.shape
    zc, yc, xc = (s // 2 for s in shape)
    slices = [
        img[zc, :, :],  # XY plane
        img[:, yc, :],  # XZ plane
        img[:, :, xc],  # YZ plane
    ]

    # Plot
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        axs[i].imshow(slices[i], vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)

    plt.show()
    plt.close(fig)


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
        center[0],
        center[1],
        center[2],
        std,
        std,
        std,
        np.max(img_patch),
        np.min(img_patch),
    )
    return fit(img_patch, gaussian_3d, initial_guess)


def fit_rotated_gaussian_3d(img_patch):
    center = [s // 2 for s in img_patch.shape]
    initial_guess = (
        center[0],
        center[1],
        center[2],
        1e-2,
        0,
        0,
        1e-2,
        0,
        1e-2,
        np.max(img_patch),
        np.min(img_patch),
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
        a11 * dx**2
        + 2 * a12 * dx * dy
        + 2 * a13 * dx * dz
        + a22 * dy**2
        + 2 * a23 * dy * dz
        + a33 * dz**2
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
        a11 * dx**2
        + 2 * a12 * dx * dy
        + 2 * a13 * dx * dz
        + a22 * dy**2
        + 2 * a23 * dy * dz
        + a33 * dz**2
    )
    return (quad <= threshold).reshape(shape)


# --- Helpers ---
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
        indexing="ij",
    )
    return np.stack(grid, axis=-1).reshape(-1, 3)


def get_center_slices(center, shape):
    """
    Gets the start and end indices of image patch to be read.

    Parameters
    ----------
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    start = [int(c - d // 2) for c, d in zip(center, shape)]
    slices = tuple(slice(s, s + d) for s, d in zip(start, shape))
    return (0, 0, *slices)


def get_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Path to image

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if ".zarr" in img_path:
        return "zarr"
    elif ".n5" in img_path:
        return "n5"
    raise ValueError(f"Unsupported image format: {img_path}")


def get_slices(voxel, shape):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    voxel : Tuple[int]
        Start voxel of the slices.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    slices = tuple(slice(v, v + d) for v, d in zip(voxel, shape))
    return (0, 0, *slices)


def get_storage_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Image path to be checked.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if util.is_s3_path(img_path):
        return "s3"
    elif util.is_gcs_path(img_path):
        return "gcs"
    else:
        raise ValueError(f"Unsupported path type: {img_path}")


def is_inbounds(shape, voxel, margin=0):
    """
    Check if voxel is contained in 3D image, with a specified margin.

    Parameters
    ----------
    shape : ArrayLike
        Shape of the 3D image.
    voxel : Tuple[int]
        Voxel coordinate to be checked.
    margin : int, optional
        Margin distance from the edges of the image. Default is 0

    Returns
    -------
    bool
        True if the voxel is contained in image, and False otherwise.
    """
    voxel = np.array(voxel)
    shape = np.array(shape)
    return bool(np.all(voxel >= margin) and np.all(voxel <= shape - margin))


def normalize(img, percentiles=(1, 99.5)):
    """
    Normalizes an image using a percentile-based scheme and clips values to
    [0, 1].

    Parameters
    ----------
    img : numpy.ndarray
        Image to be normalized.
    percentiles : Tuple[float], optional
        Upper and lower percentiles used to normalize the given image. Default
        is (1, 99.5).

    Returns
    -------
    img : numpy.ndarray
        Normalized image.
    """
    mn, mx = np.percentile(img, percentiles)
    return np.clip((img - mn) / (mx - mn + 1e-5), 0, 1)
