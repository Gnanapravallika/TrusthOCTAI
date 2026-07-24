"""PyTorch Dataset wrapper and patient-level splitting utilities for Retinal OCT."""

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically standardize dataframe image path and label column names."""
    df = df.copy()
    col_lower = {col.lower(): col for col in df.columns}

    # Find path column
    path_col = None
    for cand in ['image_path', 'filepath', 'path', 'filename', 'file']:
        if cand in col_lower:
            path_col = col_lower[cand]
            break

    # Find label column
    label_col = None
    for cand in ['label', 'class', 'target', 'category', 'condition']:
        if cand in col_lower:
            label_col = col_lower[cand]
            break

    if path_col and path_col != 'image_path':
        df['image_path'] = df[path_col]
    if label_col and label_col != 'label':
        df['label'] = df[label_col]

    # Map string labels to numeric index if needed
    if df['label'].dtype == object or isinstance(df['label'].iloc[0], str):
        df['label'] = df['label'].astype(str).str.upper().map(CLASS_TO_IDX).fillna(-1).astype(int)

    # Extract patient ID from filename to prevent data leakage (e.g. DME-123456-1.jpeg -> patient_id: DME-123456)
    if 'patient_id' not in df.columns:
        def extract_patient_id(path_str):
            fname = os.path.basename(str(path_str))
            match = re.match(r'^([A-Z0-9]+-\d+)', fname, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            return fname.split('-')[0].upper()
        df['patient_id'] = df['image_path'].apply(extract_patient_id)

    return df


def patient_level_split(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Perform patient-level split (GroupShuffleSplit) preventing patient overlap across splits."""
    df = auto_detect_columns(df)

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_outer.split(df, groups=df['patient_id']))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_relative_ratio, random_state=seed)
    train_idx, val_idx = next(gss_inner.split(train_val_df, groups=train_val_df['patient_id']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Patient-Level Split Completed:")
    print(f"  Train: {len(train_df):6d} images ({train_df['patient_id'].nunique():4d} unique patients)")
    print(f"  Val  : {len(val_df):6d} images ({val_df['patient_id'].nunique():4d} unique patients)")
    print(f"  Test : {len(test_df):6d} images ({test_df['patient_id'].nunique():4d} unique patients)")

    return train_df, val_df, test_df


class DataFrameOCTDataset(Dataset):
    """PyTorch Dataset loading OCT images from pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.loc[idx, 'image_path']
        label = int(self.df.loc[idx, 'label'])

        try:
            # Read image as RGB numpy array
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            # Fallback for unreadable images
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)
