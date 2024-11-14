"""
Scripts that runs the Code Ocean capsule
"""

import os
from pathlib import Path

from puncta_detection.detect import z1_puncta_detection
from puncta_detection.utils import utils


def run():
    """
    Run function
    """

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))
    # SCRATCH_FOLDER = Path(os.path.abspath("../scratch"))
    DATA_FOLDER = Path(os.path.abspath("../data"))

    spot_dict_path = list(DATA_FOLDER.glob("spot_channel_*.json"))
    
    if not len(spot_dict_path):
        raise FileNotFoundError("No spot channel dictionary was found!")

    spot_dict = utils.read_json_as_dict(spot_dict_path[0])
    spot_channel = spot_dict.get("spot_channels")

    if spot_channel is None:
        raise ValueError("Please, provide a spot channel in the dictionary")

    # Data
    data_channels = list(DATA_FOLDER.glob(f"*{spot_channel}.zarr"))
    segmentation_paths = list(DATA_FOLDER.glob("segmentation_*.zarr"))
    
    if len(data_channels) and len(segmentation_paths):
        data_path = data_channels[0]
        segmentation_path = segmentation_paths[0]

        output_folder = RESULTS_FOLDER.joinpath(f"{data_path.stem}_spots")
        utils.create_folder(dest_dir=str(output_folder), verbose=True)

        logger = utils.create_logger(output_log_path=str(output_folder))

        logger.info(f"Processing dataset {data_path} with segmentation {segmentation_path}")
        # Puncta detection parameters

        sigma_zyx = [1.8, 1.0, 1.0]
        background_percentage = 25
        axis_pad = int(1.6 * max(max(sigma_zyx[1:]), sigma_zyx[0]) * 5)
        min_zyx = [3, 3, 3]
        filt_thresh = 20
        raw_thresh = 180
        context_radius = 3
        radius_confidence = 0.05

        # Data loader params
        puncta_params = {
            "dataset_path": str(data_path),
            "segmentation_mask_path": str(segmentation_path),
            "multiscale": "0",
            "prediction_chunksize": (128, 128, 128),
            "target_size_mb": 2048,
            "n_workers": 0,
            "batch_size": 1,
            "axis_pad": axis_pad,
            "output_folder": output_folder,
            "logger": logger,
            "super_chunksize": None,
            "spot_parameters": {
                "sigma_zyx": sigma_zyx,
                "background_percentage": background_percentage,
                "min_zyx": min_zyx,
                "filt_thresh": filt_thresh,
                "raw_thresh": raw_thresh,
                "context_radius": context_radius,
                "radius_confidence": radius_confidence,
            },
        }

        logger.info(
            f"Dataset path: {puncta_params['dataset_path']} - Puncta detection params: {puncta_params}"
        )

        z1_puncta_detection(**puncta_params)

    else:
        raise FileNotFoundError("There are no image channels or segmentation data inside of the data folder.")

if __name__ == "__main__":
    # cProfile.run('main()', filename="/results/compute_costs.dat")
    run()
