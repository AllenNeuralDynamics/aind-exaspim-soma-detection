""" top level run script """

import os
from pathlib import Path

from puncta_detection.detect import z1_puncta_detection
from puncta_detection.utils import utils


def create_folder(dest_dir, verbose=False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def main():
    """Runs large-scale cell segmentation"""

    # Code ocean folders
    results_folder = os.path.abspath("../results")
    data_folder = os.path.abspath("../data")
    scratch_folder = os.path.abspath("../scratch")

    folders_to_process = [
        p.name
        for p in Path(f"{data_folder}/hcr_kevins_data_upsampled_masks_r1").glob("HCR*")
    ]

    print(f"Folders to process: {folders_to_process}")
    channels_to_process = ["561", "638"]

    # Puncta detection parameters
    sigma_zyx = [1.8, 1.0, 1.0]
    background_percentage = 25
    axis_pad = int(1.6 * max(max(sigma_zyx[1:]), sigma_zyx[0]) * 5)
    min_zyx = [3, 3, 3]
    filt_thresh = 20
    raw_thresh = 180
    context_radius = 3
    radius_confidence = 0.05

    ignore_list = []  # "HCR_744360-ROI-N1_2024-09-05_14-00-00"

    for folder in folders_to_process:

        print(f"PROCESSING CELLS OF {folder}")
        raw_folder = Path(data_folder).joinpath(folder)

        if raw_folder.name in ignore_list or not raw_folder.exists():
            print(
                f"Ignoring {raw_folder.name}. Check the ignore list or if the folder exists."
            )

        else:
            curr_results = Path(results_folder).joinpath(f"{raw_folder.name}_puncta")

            print(f"Raw folder {raw_folder} - {curr_results}")

            for chn in channels_to_process:
                channel_paths = list(
                    raw_folder.joinpath("corrected.ome.zarr").glob(f"*{chn}.zarr")
                )
                if len(channel_paths) == 1:

                    # Channel zarr path
                    channel_zarr_path = channel_paths[0]

                    # Segmentation path
                    segmentation_path = Path(data_folder).joinpath(
                        f"hcr_kevins_data_upsampled_masks_r1/{folder}/segmentation_mask.zarr"
                    )
                    # Path(data_folder).joinpath(f"{folder}/transform/transformed_masks.zarr")

                    if not segmentation_path.exists() or not channel_zarr_path.exists():
                        raise ValueError(
                            f"Problem processing channel {channel_zarr_path} with seg path: {segmentation_path}"
                        )

                    channel_results = curr_results.joinpath(chn)

                    if channel_results.joinpath("spots.npy").exists():
                        print(
                            f"Ignoring {channel_results} because it already has spots."
                        )
                    else:
                        create_folder(channel_results)

                        logger = utils.create_logger(
                            output_log_path=str(channel_results)
                        )

                        # Data loader params
                        puncta_params = {
                            "dataset_path": str(channel_zarr_path),
                            "segmentation_mask_path": str(segmentation_path),
                            "multiscale": "0",
                            "prediction_chunksize": (128, 128, 128),
                            "target_size_mb": 3048,
                            "n_workers": 0,
                            "batch_size": 1,
                            "axis_pad": axis_pad,
                            "output_folder": channel_results,
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
                    print(f"Channel {chn} not in dataset {raw_folder}")


if __name__ == "__main__":
    main()
