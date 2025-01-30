"""
Created on Wed Jan 8 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals and classifies them for a whole-brain
image dataset.

"""

from scipy.optimize import OptimizeWarning
from time import time

import warnings

from aind_exaspim_soma_detection import soma_proposal_classification as spc
from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import util

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    """
    Runs the soma proposal generation and classification pipeline for a
    whole-brain image dataset.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    print("\nBrain_ID:", brain_id)
    proposals = generate_proposals()
    accepted_proposals = classify_proposals(proposals)
    return filter_accepts(accepted_proposals)


def generate_proposals():
    """
    Generates soma proposals and saves the results if applicable.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # Main
    t0 = time()
    print("\nSteps 1-2: Generate and Filter Proposals")
    img_prefix = img_prefixes[brain_id] + str(multiscale_1)
    proposals = spg.generate_proposals(
        img_prefix,
        overlap,
        multiscale_1,
        patch_shape_1,
        bright_threshold=bright_threshold,
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    print("\n# Proposals Generated:", len(proposals))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_proposals:
        util.write_points(
            f"/root/capsule/results/{brain_id}/proposals",
            proposals,
            color="0.0 1.0 0.0",
            prefix="proposal_",
            radius=15,
        )
    return proposals


def classify_proposals(proposals):
    """
    Classifies a list of soma proposals and saves the results if applicable.

    Parameters
    ----------
    proposals : List[Tuple[float]]
        List of proposals, where each is represented by an xyz coordinate.

    Returns
    -------
    None

    """
    # Main
    t0 = time()
    print("\nStep 3: Classify Proposals")
    img_prefix = img_prefixes[brain_id] + str(multiscale_2)
    accepted_proposals = spc.classify_proposals(
        brain_id,
        proposals,
        img_prefix,
        model_path,
        multiscale_2,
        patch_shape_2,
        threshold,
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    print("\n# Somas Detected:", len(accepted_proposals))
    print("% Proposals Accepted:", len(accepted_proposals) / len(proposals))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_accepts:
        util.write_points(
            f"/root/capsule/results/{brain_id}/accepts",
            accepted_proposals,
            color="0.0 0.0 1.0",
            prefix="accept_",
            radius=20,
        )
    return accepted_proposals


def filter_accepts(accepted_proposals):
    """
    Filters a list of accpeted soma proposals and saves the results if
    applicable.

    Parameters
    ----------
    accepted_proposals : List[Tuple[float]]
        List of accepted proposals, where each is represented by an xyz
        coordinate.

    Returns
    -------
    None

    """
    # Main
    t0 = time()
    print("\nStep 4: Filter Accepted Proposals")
    img_prefix = img_prefixes[brain_id] + str(multiscale_3)
    filtered_accepts = spc.branchiness_filtering(
        img_prefix, accepted_proposals, patch_shape_3
    )
    t, unit = util.time_writer(time() - t0)

    # Report results
    n = len(filtered_accepts)
    print("\n# Somas Detected:", n)
    print("% Proposals Accepted:", n / len(accepted_proposals))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_filtered_accepts:
        util.write_points(
            f"/root/capsule/results/{brain_id}/filtered_accepts",
            filtered_accepts,
            color="1.0 0.0 0.0",
            prefix="soma_",
            radius=25,
        )


if __name__ == "__main__":
    # Parameters
    brain_id = "730902"
    save_proposals = True
    save_accepts = True
    save_filtered_accepts = True

    # Parameters - Proposal Generation
    multiscale_1 = 4
    patch_shape_1 = (64, 64, 64)
    bright_threshold = 120
    overlap = (28, 28, 28)

    # Parameters - Proposal Classification
    multiscale_2 = 1
    patch_shape_2 = (102, 102, 102)
    threshold = 0.9
    model_path = "/root/capsule/data/benchmarked_models/model_v1_cosine-sch_f1=0.9667.pth"

    # Parameters - Accpeted Proposal Filtering
    multiscale_3 = 2
    patch_shape_3 = (48, 48, 48)

    # Initializations
    prefix_lookup_path = "/root/capsule/data/exaspim_image_prefixes.json"
    img_prefixes = util.read_json(prefix_lookup_path)
    util.mkdir(f"/root/capsule/results/{brain_id}", delete=True)

    # Main
    main()
