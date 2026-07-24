"""Dataset scanning and automatic mapping utilities for Kermany OCT dataset."""

import os
import glob
import pandas as pd
from oct_datasets.dataset import CLASS_TO_IDX


def scan_kermany_dataset(data_dir: str = "/content/OCT2017") -> pd.DataFrame:
    """Recursively scan data directory and build structured DataFrame mapping."""
    records = []
    supported_exts = ('.jpeg', '.jpg', '.png')

    # List of candidate paths to search if data_dir is missing or empty
    candidate_paths = [
        data_dir,
        "/content/Kermany",
        "/content/OCT2017",
        "/content/OCT2017/OCT2017 ",
        "/content/Kermany/OCT2017 ",
        "/content/kermany2017/OCT2017 ",
        "/content/datasets/Kermany/OCT2017",
        "/content/drive/MyDrive/OCT2017",
        "/content"
    ]

    # Include kagglehub cache locations
    kagglehub_matches = glob.glob("/root/.cache/kagglehub/datasets/**/OCT2017*", recursive=True)
    candidate_paths.extend(kagglehub_matches)

    target_dir = None
    for cand in candidate_paths:
        if os.path.exists(cand):
            # Check if directory contains images or image subfolders
            sub_files = glob.glob(f"{cand}/**/*.jpeg", recursive=True) + glob.glob(f"{cand}/**/*.jpg", recursive=True) + glob.glob(f"{cand}/**/*.png", recursive=True)
            if len(sub_files) > 0:
                target_dir = cand
                break

    if target_dir is None:
        target_dir = data_dir

    print(f"Scanning OCT dataset at: {target_dir}")

    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(supported_exts):
                folder_name = os.path.basename(root).strip().upper()
                if folder_name in CLASS_TO_IDX:
                    full_path = os.path.join(root, f)
                    records.append({
                        "image_path": full_path,
                        "label": CLASS_TO_IDX[folder_name],
                        "class_name": folder_name
                    })

    if len(records) == 0:
        raise ValueError(
            f"❌ No OCT scans found at {target_dir}!\n"
            "Please check Section 2 dataset download step to ensure files were extracted."
        )

    df = pd.DataFrame(records)
    print(f"Successfully cataloged {len(df):,} total OCT scans across 4 classes.")
    return df
