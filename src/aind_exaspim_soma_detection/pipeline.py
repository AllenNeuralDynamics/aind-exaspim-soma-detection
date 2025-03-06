"""
Created on Wed Jan 8 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals and classifies them for a whole-brain
image dataset.

"""

from scipy.optimize import OptimizeWarning
from time import time

import os
import warnings

from aind_exaspim_soma_detection import soma_proposal_classification as spc
from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import util

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_pipeline(
    brain_id,
    img_prefix,
    output_dir,
    proposal_params,
    classify_params,
    filter_params=None
):
    """
    Runs the soma proposal generation and classification pipeline for a
    whole-brain image dataset.

    Parameters
    ----------
    brain_id : str
        Unique identifier for the whole-brain dataset.
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    output_dir : str
        Path to directory that results will be written to.
    proposal_params : dict
        Dictionary containing values for optional parameters used by the
        routine "generate_proposals".
    classify_params : dict
        Dictionary containing values for optional parameters used by the
        routine "classify_proposls".
    filter_params : dict, optional
        Dictionary containing values for optional parameters used by the
        routine "filter_accepts". The default is None.

    Returns
    -------
    None

    """
    # Sanity checks
    util.mkdir(output_dir, delete=True)
    model_path_exists = os.path.exists(classify_params["model_path"])
    assert model_path_exists, "model_path does not exist!"

    # Detect somas
    proposals = generate_proposals(img_prefix, **proposal_params)
    accepts = classify_proposls(img_prefix, proposals, **classify_params)
    write_results(output_dir, f"somas-{brain_id}.txt", accepts)

    # Filter detected somas (optional)
    if filter_params is not None:
        accepts = filter_accepts(img_prefix, accepts, **filter_params)
        write_results(output_dir, f"filtered-somas-{brain_id}.txt", accepts)


def generate_proposals(
    img_prefix,
    multiscale=4,
    patch_shape=(64, 64, 64),
    patch_overlap=(32, 32, 32),
    bright_threshold=0,
    output_dir=None,
    save_swcs=False
):
    """
    Generates soma proposals and saves the each proposal coordinate as an SWC
    file if applicable.

    Parameters
    ----------
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    multiscale : int, optional
        Level in the image pyramid that image patches are read from. The
        default is 4.
    patch_shape : Tuple[int], optional
        Shape of each image patch. The default is (64, 64, 64).
    patch_overlap : int, optional
        Overlap between adjacent image patches in each dimension. The default
        is (32, 32, 32).
    bright_threshold : int, optional
        Brightness threshold used to filter proposals and image patches. The
        default is 0.
    save_swcs : bool, optional
        Indication of whether to save each proposal coordinate as an SWC file.
        The default is False.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of proposals.

    """
    # Main
    t0 = time()
    print("\nSteps 1-2: Generate and Filter Proposals")
    img_prefix += str(multiscale)
    proposals = spg.generate_proposals(
        img_prefix,
        multiscale,
        patch_shape,
        patch_overlap,
        bright_threshold=bright_threshold,
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    print("\n# Proposals Generated:", len(proposals))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_swcs:
        util.write_points(
            os.path.join(output_dir, "proposals"),
            proposals,
            color="0.0 1.0 0.0",
            prefix="proposal_",
            radius=15,
        )
    return proposals


def classify_proposls(
    img_prefix,
    proposals,
    accept_threshold=0.4,
    model_path=None,
    multiscale=1,
    patch_shape=(102, 102, 102),
    save_swcs=False,
):
    """
    Classifies a list of soma proposals and saves the results if applicable.

    Parameters
    ----------
    img_prefix : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    proposals : List[Tuple[float]]
        List of proposals, where each is represented by an xyz coordinate.
    accept_threshold : float, optional
        Threshold applied to model predictions, above which a proposal is
        classified as a soma. The default is 0.4.
    model_path : str
        Path to the pre-trained model that is used to classify the proposals.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.
    patch_shape : tuple of int
        Shape of image patches to be used for inference.
    save_swcs : bool, optional
        Indication of whether to save each proposal coordinate as an SWC file.
        The default is False.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of accepted proposals.

    """
    # Main
    t0 = time()
    print("\nStep 3: Classify Proposals")
    img_prefix += str(multiscale)
    accepts = spc.classify_proposals(
        -1,
        proposals,
        img_prefix,
        model_path,
        multiscale,
        patch_shape,
        accept_threshold,
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    print("\n# Proposals Accepted:", len(accepts))
    print("% Proposals Accepted:", round(len(accepts) / len(proposals), 4))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_swcs:
        util.write_points(
            os.path.join(output_dir, "accepts"),
            accepts,
            color="0.0 0.0 1.0",
            prefix="accept_",
            radius=20,
        )
    return accepts


def filter_accepts(
    img_prefix,
    accepts,
    multiscale=3,
    patch_shape=(40, 40, 40),
    save_swcs=False,
):
    """
    Filters a list of accpeted soma proposals and saves the results if
    applicable.

    Parameters
    ----------
    accepts : List[Tuple[float]]
        Physical coordinates of accepted proposals.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of accepted proposals that passed filtering step.

    """
    # Main
    t0 = time()
    print("\nStep 4: Filter Accepted Proposals")
    img_prefix += str(multiscale)
    filtered_accepts = spc.branchiness_filtering(
        img_prefix, accepts, multiscale, patch_shape
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    print("# Filtered Accepts:", len(filtered_accepts))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_swcs:
        util.write_points(
            os.path.join(output_dir, "filtered_accepts"),
            filtered_accepts,
            color="1.0 0.0 0.0",
            radius=25,
        )
    return filtered_accepts


def write_results(output_dir, filename, coords_list):
    """
    Writes a list of xyz coordinates to a txt file.

    Parameters
    ----------
    output_dir : str
        Path to directory that results will be written to.
    filename : str
        Name of txt file to be written.
    xyz_list : List[Tuple[float]]
        List of 3D coordinates to write to txt file.

    Returns
    -------
    None

    """
    path = os.path.join(output_dir, filename)
    util.write_to_list(path, coords_list)


if __name__ == "__main__":
    # Initializations
    root_dir = "/root/capsule/data"
    model_path = f"{root_dir}/benchmarked_models/model_v1_cosine-sch_f1=0.9667.pth"
    img_prefixes = util.read_json(f"{root_dir}/exaspim_image_prefixes.json")

    # Parameters
    brain_id = "709222"
    img_prefix = img_prefixes[brain_id]
    output_dir = f"/root/capsule/scratch/soma-detection-{brain_id}"
    proposal_params = {
        "multiscale": 4,
        "patch_shape": (64, 64, 64),
        "bright_threshold": 150,
        "patch_overlap":(28, 28, 28),
        "save_swcs": False,
    }
    classify_params = {
        "multiscale": 1,
        "patch_shape": (102, 102, 102),
        "accept_threshold": 0.4,
        "model_path": model_path,
        "save_swcs": False,
    }
    filter_params = {
        "multiscale": 3,
        "patch_shape": (40, 40, 40),
        "save_swcs": False,
    }

    # Main
    run_pipeline(
        brain_id,
        img_prefix,
        output_dir,
        proposal_params,
        classify_params,
        filter_params
    )
