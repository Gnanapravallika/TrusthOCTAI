"""Main training script for TrustOCT models."""

import os
import argparse
import torch
import pandas as pd
from oct_datasets.utils import scan_kermany_dataset
from oct_datasets.dataset import patient_level_split
from engine.trainer import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train TrustOCT framework models.")
    parser.add_argument("--data_dir", type=str, default="datasets/Kermany/OCT2017", help="Path to Kermany OCT2017 dataset directory.")
    parser.add_argument("--model_name", type=str, default="msf_cbam_resnet50", help="Model experiment name (resnet50, msf_resnet50, msf_cbam_resnet50).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

    args = parser.parse_args()

    # Step 1: Scan dataset and perform patient-level split
    df = scan_kermany_dataset(args.data_dir)
    train_df, val_df, test_df = patient_level_split(df)

    # Step 2: Run experiment training
    exp_dir = run_experiment(
        model_name=args.model_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )

    print(f"\n✅ Experiment {args.model_name} completed! Weights saved to: {exp_dir}")


if __name__ == "__main__":
    main()
