"""
Created on Thu Apr 30 18:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for working with SWC files.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from tqdm import tqdm

import os
import pandas as pd

from aind_exaspim_soma_detection.utils import util


def load_dataset_examples(bucket_name, prefix):
    examples = list()
    brain_ids = util.list_gcs_subdirs(bucket_name, prefix)
    bucket = storage.Client().bucket(bucket_name)
    for brain_id in tqdm(brain_ids, desc="Load Data"):
        examples.extend(load_brain_examples(bucket, prefix, brain_id))
    return pd.DataFrame(examples)


def load_brain_examples(bucket, prefix, brain_id):
    for label_str, label in [("accepts", 1), ("rejects", 0)]:
        subprefix = f"{prefix}/{brain_id}/{label_str}/"
        with ThreadPoolExecutor(max_workers=64) as executor:
            # Assign threads
            threads = list()
            for blob in bucket.list_blobs(prefix=subprefix):
                if blob.name.endswith(".swc"):
                    threads.append(
                        executor.submit(_load_blob, blob, brain_id, label)
                    )

            # Compile results
            examples = list()
            for thread in as_completed(threads):
                result = thread.result()
                if result is not None:
                    examples.append(result)
    return examples


def parse_swc_point(content, source=""):
    """
    Parses a single-point SWC file to extract a single coordinate.
    """
    offset = (0.0, 0.0, 0.0)
    for line in content.splitlines():
        # Check if line is empty
        line = line.strip()
        if not line:
            continue

        # Check for commented lines
        if line.startswith("#"):
            tokens = line.lstrip("#").strip().split()
            if line.startswith("# OFFSET"):
                offset = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            continue

        # Get coordinate
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed SWC line in {source!r}: {line!r}")
        _, _, x, y, z, *_ = parts
        return (
            float(x) + offset[0],
            float(y) + offset[1],
            float(z) + offset[2],
        )

    raise ValueError(f"No data rows found in SWC file: {source!r}")


def _load_blob(blob, brain_id, label):
    filename = blob.name.split("/")[-1]
    try:
        content = blob.download_as_text()
        x, y, z = parse_swc_point(content, source=blob.name)
        example = {
            "brain_id": brain_id,
            "label": label,
            "swc_filename": filename,
            "x": x,
            "y": y,
            "z": z,
        }
        return example
    except ValueError as e:
        print(f"  WARNING: {e} — skipping")
        return None


# --- Miscellaneous ---
def split_swc_into_points(input_swc_path, output_dir):
    """
    Reads an SWC file and writes each point as its own SWC file.

    Parameters
    ----------
    input_swc_path : str
        Path to the input SWC file.
    output_dir : str
        Directory where individual SWC files will be saved.
    """
    # Read SWC file
    lines = util.read_txt(input_swc_path).splitlines()

    # Extract offset header if present
    offset_header = None
    for line in lines:
        if line.startswith("# OFFSET"):
            offset_header = line.strip()
            break

    # Process SWC file
    lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
    os.makedirs(output_dir, exist_ok=True)
    for i, line in enumerate(lines):
        # Extract content
        parts = line.strip().split()
        node_id = parts[0]
        parts[0] = "1"
        new_line = " ".join(parts) + "\n"

        # Write SWC file
        output_path = os.path.join(output_dir, f"{node_id}.swc")
        with open(output_path, "w") as out:
            if offset_header:
                out.write(f"{offset_header}\n")
            out.write(new_line)

    print(f"Saved {len(lines)} SWC files to {output_dir}")
