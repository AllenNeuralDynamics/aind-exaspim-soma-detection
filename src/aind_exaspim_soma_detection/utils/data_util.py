"""
Created on Thu Apr 30 18:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for loading positive and negative soma examples.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import pandas as pd
import random

from aind_exaspim_soma_detection.utils import util


def load_dataset_examples(bucket_name, prefix):
    examples = list()
    brain_ids = util.list_gcs_subdirs(bucket_name, prefix)
    for brain_id in tqdm(brain_ids, desc="Load Data"):
        if brain_id not in ["719654", "789202", "802450"]:
            examples.extend(
                load_brain_examples(bucket_name, prefix, brain_id)
            )
    return pd.DataFrame(examples)


def load_brain_examples(bucket_name, prefix, brain_id):
    examples = list()
    for label_str, label in [("accepts", 1), ("rejects", 0)]:
        subprefix = f"{prefix}/{brain_id}/{label_str}"
        with ThreadPoolExecutor(max_workers=128) as executor:
            # Extract filenames to load
            filenames = util.list_gcs_subdirs(bucket_name, subprefix)
            if brain_id == "802449":
                n = len(filenames) // 2
                filenames = random.sample(filenames, n)

            # Assign threads
            threads = list()
            for filename in filenames:
                if filename.endswith(".swc"):
                    gcs_path = f"gs://{bucket_name}/{subprefix}/{filename}"
                    threads.append(
                        executor.submit(_load_example, gcs_path, brain_id, label)
                    )

            # Compile results
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


def _load_example(gcs_path, brain_id, label):
    try:
        content = util.read_txt(gcs_path)
        filename = gcs_path.split("/")[-1]
        xyz = parse_swc_point(content, source=gcs_path)
        example = {
            "brain_id": brain_id,
            "label": label,
            "swc_filename": filename,
            "xyz": xyz,
        }
        return example
    except ValueError as e:
        print(f"  WARNING: {e} — skipping")
        return None


# --- Helpers ---
def partition_dataset(df, train_frac=0.6):
    """
    Partitions DataFrame indices into train, val, and test splits.

    Parameters
    ----------
    df : pd.DataFrame
    train_frac : float

    Returns
    -------
    dict with keys "train", "val", "test" mapping to lists of df indices.
    """
    assert train_frac > 0 and train_frac < 1
    idx = df.index.tolist()
    train_idx, val_idx = train_test_split(idx, train_size=train_frac)
    return {"train": train_idx, "val": val_idx}


def split_swc_into_points(input_swc_path, output_dir):
    """
    Reads an SWC file containing single points and writes each point as its
    own SWC file.

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


def summarize_dataset(df):
    summary = (
        df.groupby(["brain_id", "label"])
        .size()
        .unstack(fill_value=0)
    )
    summary.columns = [("accepts" if c else "rejects") for c in summary.columns]
    for col in ["accepts", "rejects"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["total"] = summary["accepts"] + summary["rejects"]
    return summary
