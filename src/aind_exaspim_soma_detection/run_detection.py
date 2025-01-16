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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    # Part 1: Generate Soma Proposals
    t0 = time()
    print("\nPart 1: Generate Soma Proposals")
    img_prefix = img_prefixes[brain_id] + str(multiscale_1)
    proposals = spg.generate_proposals(
        img_prefix,
        overlap,
        multiscale_1,
        patch_shape_1,
        bright_threshold=bright_threshold,
    )
    t, unit = util.time_writer(time() - t0)
    print("\n# Proposals Generated:", len(proposals))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_proposals_bool:
        output_dir = "/root/capsule/results/proposals"
        save_result(proposals, output_dir, "0.0 0.0 1.0", "proposal_", 20)

    # Part 2: Classify Soma Proposals
    t0 = time()
    print("\nPart 2: Classify Soma Proposals")
    img_prefix = img_prefixes[brain_id] + str(multiscale_2)
    somas = spc.classify_proposals(
        brain_id,
        proposals,
        img_prefix,
        model_path,
        multiscale_2,
        patch_shape_2,
        threshold,
        
    )
    t, unit = util.time_writer(time() - t0)
    print("\n# Somas Detected:", len(somas))
    print("% Proposals Accepted:", len(somas) / len(proposals))
    print(f"Runtime: {round(t, 4)} {unit}")
    if save_proposals_bool:
        output_dir = "/root/capsule/results/somas"
        save_result(somas, output_dir, "1.0 0.0 0.0", "soma_", 25)


def save_result(xyz_list, output_dir, color, prefix, radius):
    util.write_points(
        output_dir, xyz_list, color=color, prefix=prefix, radius=radius,
    )


if __name__ == "__main__":
    # Parameters
    brain_id = "721830"
    save_proposals_bool = True
    save_somas_bool = True

    # Parameters ~ Proposal Generation
    multiscale_1 = 4
    patch_shape_1 = (64, 64, 64)
    bright_threshold = 120
    overlap = (28, 28, 28)

    # Parameters ~ Proposal Classification
    multiscale_2 = 1
    patch_shape_2 = (102, 102, 102)
    threshold = 0.3
    model_path = "/root/capsule/scratch/soma_classifiers/soma_classifiers_2025-01-11 21:07:42.683209/model_8640_f1=0.93.pth"

    # Initializations
    prefix_lookup_path = "/root/capsule/data/exaspim_image_prefixes.json"
    img_prefixes = util.read_json(prefix_lookup_path)

    # Main
    print("\nBrain_ID:", brain_id)
    main()
