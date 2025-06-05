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
import pandas as pd
import warnings

from aind_exaspim_soma_detection import soma_proposal_classification as spc
from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import util

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_pipeline(
    brain_id,
    img_path,
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
    img_path : str
        Prefix (or path) of a whole brain image stored in a S3 bucket.
    output_dir : str
        Path to directory that results are written to.
    proposal_params : dict
        Dictionary containing values for optional parameters used by the
        routine "generate_proposals".
    classify_params : dict
        Dictionary containing values for optional parameters used by the
        routine "classify_proposals".
    filter_params : dict, optional
        Dictionary containing values for optional parameters used by the
        routine "quantify_accepts". The default is None.

    Returns
    -------
    None

    """
    # Initializations
    t0 = time()
    util.mkdir(output_dir, delete=True)
    model_path = classify_params["model_path"]
    update_log(output_dir, f"Brain_ID: {brain_id}")
    update_log(output_dir, f"Image Prefix: {img_path}")
    update_log(output_dir, f"Model Name: {model_path}")
    model_path_exists = os.path.exists(model_path)
    assert model_path_exists, "model_path does not exist!"

    # Detect somas
    proposals = generate_proposals(img_path, **proposal_params)
    accepts = classify_proposals(img_path, proposals, **classify_params)

    # Compute soma metrics
    accepts_df = quantify_accepts(img_path, accepts, **filter_params)
    path = os.path.join(output_dir, f"somas-{brain_id}.csv")
    pd.DataFrame(accepts_df).to_csv(path, index=False)

    # Report runtime
    t, unit = util.time_writer(time() - t0)
    update_log(output_dir, f"\nTotal Runtime: {round(t, 4)} {unit}")


def generate_proposals(
    img_path,
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
    img_path : str
        Path to whole brain image stored in a S3 bucket.
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
    output_dir : str, optional
        Path to directory that results are written to. The default is None.
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
    update_log(output_dir, "\nSteps 1: Generate Proposals")
    img_path += str(multiscale)
    proposals = spg.generate_proposals(
        img_path,
        multiscale,
        patch_shape,
        patch_overlap,
        bright_threshold=bright_threshold,
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    update_log(output_dir, f"# Proposals: {len(proposals)}")
    update_log(output_dir, f"Runtime: {round(t, 4)} {unit}")
    if save_swcs:
        util.write_points(
            os.path.join(output_dir, "proposals.zip"),
            proposals,
            color="0.0 1.0 0.0",
            prefix="proposal_",
            radius=15,
        )
    return proposals


def classify_proposals(
    img_path,
    proposals,
    accept_threshold=0.4,
    model_path=None,
    multiscale=1,
    patch_shape=(102, 102, 102),
    output_dir=None,
    save_swcs=False,
):
    """
    Classifies a list of soma proposals and saves the results if applicable.

    Parameters
    ----------
    img_path : str
        Path to whole brain image stored in a S3 bucket.
    proposals : List[Tuple[float]]
        List of proposals, where each is represented by an xyz coordinate.
    accept_threshold : float, optional
        Threshold applied to model predictions, above which a proposal is
        classified as a soma. The default is 0.4.
    model_path : str, optional
        Path to the pre-trained model that is used to classify the proposals.
        The default is None.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        The default is 1.
    patch_shape : Tuple[int], optional
        Shape of image patches to be used for inference. The default is
        (102, 102, 102).
    output_dir : str, optional
        Path to directory that results are written to. The default is None.
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
    update_log(output_dir, "\nStep 2: Classify Proposals")
    img_path += str(multiscale)
    accepts = spc.classify_proposals(
        -1,
        proposals,
        img_path,
        model_path,
        multiscale,
        patch_shape,
        accept_threshold,
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    percent_accept = round(len(accepts) / len(proposals), 4)
    update_log(output_dir, f"# Accepts: {len(accepts)}")
    update_log(output_dir, f"% Accepted: {percent_accept}")
    update_log(output_dir, f"Runtime: {round(t, 4)} {unit}")
    if save_swcs:
        util.write_points(
            os.path.join(output_dir, "accepts.zip"),
            accepts,
            color="0.0 0.0 1.0",
            prefix="accept_",
            radius=20,
        )
    return accepts


def quantify_accepts(
    img_path,
    accepts,
    multiscale=3,
    patch_shape=(42, 42, 42),
    output_dir=None,
    save_swcs=False,
):
    """
    Filters a list of accpeted soma proposals and saves the results if
    applicable.

    Parameters
    ----------
    img_path : str
        Path to whole brain image stored in a S3 bucket.
    accepts : List[Tuple[float]]
        Physical coordinates of accepted proposals.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        The default is 3.
    patch_shape : Tuple[int]
        Shape of image patches to be used. The default is (40, 40, 40).
    output_dir : str, optional
        Path to directory that results are written to. The default is None.
    save_swcs : bool, optional
        Indication of whether to save each proposal coordinate as an SWC file.
        The default is False.

    Returns
    -------
    List[Tuple[float]]
        Physical coordinates of accepted proposals that passed filtering step.

    """
    # Main
    t0 = time()
    update_log(output_dir, "\nStep 3: Filter Accepted Proposals")
    img_path += str(multiscale)
    filtered_accepts_df = spc.compute_metrics(
        img_path, accepts, multiscale, patch_shape
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    update_log(output_dir, f"# Filtered Accepts: {len(filtered_accepts_df)}")
    update_log(output_dir, f"Runtime: {round(t, 4)} {unit}")
    if save_swcs:
        util.write_points(
            os.path.join(output_dir, "filtered_accepts.zip"),
            filtered_accepts_df["xyz"],
            color="1.0 0.0 0.0",
            radius=25,
        )
    return filtered_accepts_df


def update_log(output_dir, log_info):
    """
    Appends log information to a specified text file.

    Parameters
    ----------
    output_dir : str
        Path to directory that results will be written to.
    log_info : str
        Information to be written to the file.

    Returns
    -------
    None

    """
    print(log_info)
    if output_dir is not None:
        path = os.path.join(output_dir, "log.txt")
        with open(path, 'a') as file:
            file.write(log_info + "\n")


if __name__ == "__main__":
    # Initializations
    root_dir = "/root/capsule/data"
    model_name = "model_v1_cosine-sch_f1=0.9667.pth"
    model_path = f"{root_dir}/benchmarked_models/{model_name}"
    img_pathes = util.read_json(f"{root_dir}/exaspim_image_prefixes.json")

    # Parameters
    brain_id = "709222"
    img_path = img_pathes[brain_id]
    output_dir = f"/root/capsule/scratch/soma-detection-{brain_id}"
    proposal_params = {
        "multiscale": 4,
        "patch_shape": (64, 64, 64),
        "bright_threshold": 150,
        "patch_overlap": (28, 28, 28),
        "output_dir": output_dir,
        "save_swcs": False,
    }
    classify_params = {
        "multiscale": 1,
        "patch_shape": (102, 102, 102),
        "accept_threshold": 0.4,
        "model_path": model_path,
        "output_dir": output_dir,
        "save_swcs": False,
    }
    filter_params = {
        "multiscale": 3,
        "patch_shape": (40, 40, 40),
        "output_dir": output_dir,
        "save_swcs": False,
    }

    # Main
    run_pipeline(
        brain_id,
        img_path,
        output_dir,
        proposal_params,
        classify_params,
        filter_params
    )
