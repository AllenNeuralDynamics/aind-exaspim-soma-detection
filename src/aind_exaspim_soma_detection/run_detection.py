"""
Created on Wed Nov 27 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that generates soma proposals.

"""

from random import sample
from time import time

import warnings

from aind_exaspim_soma_detection import soma_proposal_generation as spg
from aind_exaspim_soma_detection.utils import img_util, util

warnings.filterwarnings("ignore", category=RuntimeWarning)


def run():
    # Proposal Generation
    t0 = time()
    print("\nBrain_ID:", brain_id)
    print("\nStep 1: Soma Proposal Generation")
    proposals = spg.run_on_whole_brain(
        img_prefix,
        overlap,
        patch_shape,
        multiscale,
        bright_threshold=bright_threshold,
    )
    print("\n# Proposals Generated:", len(proposals))
    print("Runtime:", time() - t0)

    # Save Results
    output_dir = "/root/capsule/results/soma_proposals"
    util.write_points(output_dir, proposals, color="1.0 0.0 0.0", prefix="proposal_")

    proposals = sample(proposals, 200)
    output_dir = "/root/capsule/results/soma_proposals_training"
    util.write_points(output_dir, proposals, color="1.0 0.0 0.0", prefix="proposal_")


if __name__ == "__main__":
    # Parameters
    bucket_name = "aind-open-data"
    brain_id = "704522"
    multiscale = 4

    bright_threshold = 10
    overlap = (28, 28, 28)
    patch_shape = (64, 64, 64)

    # Initializations
    img_prefixes = util.read_json("/root/capsule/data/exaspim_image_prefixes.json")
    img_prefix = img_prefixes[brain_id] + str(multiscale)

    # Main
    run()
