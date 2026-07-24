"""Dataset scanning and automatic mapping utilities for Kermany OCT dataset."""

import os
import pandas as pd
from oct_datasets.dataset import CLASS_TO_IDX


def scan_kermany_dataset(data_dir: str = "datasets/Kermany/OCT2017") -> pd.DataFrame:
    """Recursively scan data directory and build structured DataFrame mapping."""
    records = []
    supported_exts = ('.jpeg', '.jpg', '.png')

    if not os.path.exists(data_dir):
        # Look in common alternative Kaggle download locations
        alt_paths = [
            "/content/OCT2017",
            "/content/OCT2017/OCT2017 ",
            "/content/kermany2017/OCT2017 ",
            "/content/datasets/Kermany/OCT2017",
            "/content/drive/MyDrive/OCT2017"
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                data_dir = alt
                break

    print(f"Scanning OCT dataset at: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(supported_exts):
                folder_name = os.path.basename(root).upper()
                if folder_name in CLASS_TO_IDX:
                    full_path = os.path.join(root, f)
                    records.append({
                        "image_path": full_path,
                        "label": CLASS_TO_IDX[folder_name],
                        "class_name": folder_name
                    })

    df = pd.DataFrame(records)
    print(f"Successfully cataloged {len(df):,} total OCT scans across 4 classes.")
    return df
