"""
Created on Wed Nov 27 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

"""

from time import time

import warnings

from aind_exaspim_soma_detection import soma_proposal_classification as spc
from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import util

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    # Part 1: Generate Soma Proposals
    t0 = time()
    print("\nPart 1: Soma Proposal Generation")
    img_prefix = img_prefixes[brain_id] + str(multiscale_1)
    proposals = spg.generate_proposals(
        img_prefix,
        overlap,
        multiscale_1,
        patch_shape_1,
        bright_threshold=bright_threshold,
    )
    print("\n# Proposals Generated:", len(proposals))
    print("Runtime:", time() - t0)
    if save_proposals_bool:
        output_dir = "/root/capsule/results/proposals"
        save_result(proposals, output_dir, "proposal_")

    # Part 2: Classify Soma Proposals
    t0 = time()
    print("\nPart 2: Soma Proposal Classification")
    img_prefix = img_prefixes[brain_id] + str(multiscale_1)
    somas = spc.classify_proposals(
        brain_id,
        proposals,
        img_prefix,
        multiscale_2,
        patch_shape_2,
        confidence_threshold,
    )
    print("\n# Proposals Generated:", len(proposals))
    print("Runtime:", time() - t0)
    if save_proposals_bool:
        output_dir = "/root/capsule/results/somas"
        save_result(somas, output_dir, "soma_")


def save_result(xyz_list, output_dir, prefix):
    util.write_points(
        output_dir, xyz_list, color="1.0 0.0 0.0", prefix=prefix
    )


if __name__ == "__main__":
    # Parameters
    bucket_name = "aind-open-data"
    brain_id = "704522"
    save_proposals_bool = True
    save_somas_bool = True

    # Parameters ~ Proposal Generation
    multiscale_1 = 4
    patch_shape_1 = (64, 64, 64)
    bright_threshold = 10
    overlap = (28, 28, 28)

    # Parameters ~ Proposal Classification
    multiscale_2 = 1
    patch_shape_2 = (102, 102, 102)
    confidence_threshold = 0.4
    model_path = None

    # Initializations
    prefix_lookup_path = "/root/capsule/data/exaspim_image_prefixes.json"
    img_prefixes = util.read_json(prefix_lookup_path)

    # Main
    print("\nBrain_ID:", brain_id)
    main()
