"""Main entry point for training, evaluating, and visualizing TrustOCT framework experiments."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
import numpy as np
import torch
import yaml

# Add project root to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from oct_datasets.dataset import get_dataset_and_loader, verify_dataset, generate_statistics_report
from models.trustoct import build_model
from engine.trainer import Trainer
from engine.tester import test_model
from evaluation.metrics import calculate_classification_metrics, plot_confusion_matrix
from evaluation.calibration import calculate_ece, calculate_brier_score, plot_reliability_diagram
from evaluation.report import save_reports, evaluate_cross_dataset
from explainability.visualization import compare_and_save_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="TrustOCT Framework Orchestrator")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Path to model architecture configuration YAML."
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train.yaml",
        help="Path to training loop configuration YAML."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="EXP006_TrustOCT",
        help="Name of the active experiment folder."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train_eval", "eval_only", "verify"],
        help="Execution mode."
    )
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[OK] Random seeds locked globally to: {seed}")


def main():
    args = parse_args()
    
    # 0. Load configs
    with open(args.train_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(args.dataset_config, "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Apply global seeds
    set_seed(train_cfg.get("seed", 42))

    experiment_dir = os.path.join(project_root, "outputs", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Mode: Verify
    if args.mode == "verify":
        report_path = os.path.join(experiment_dir, "dataset_report.json")
        verify_dataset(args.dataset_config, report_path)
        stats_path = os.path.join(experiment_dir, "dataset_statistics.json")
        generate_statistics_report(args.dataset_config, stats_path)
        return

    # 1. Build Data Loaders
    print("Building datasets and loaders...")
    train_dataset, train_loader = get_dataset_and_loader("train", args.dataset_config, args.train_config)
    val_dataset, val_loader = get_dataset_and_loader("val", args.dataset_config, args.train_config)
    test_dataset, test_loader = get_dataset_and_loader("test", args.dataset_config, args.train_config)

    # 2. Build Model
    print("Assembling TrustOCT model components...")
    model = build_model(args.model_config)

    # 3. Train Model (if requested)
    if "train" in args.mode:
        print("Compiling training engine...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_config_path=args.train_config,
            model_config_path=args.model_config,
            experiment_dir=experiment_dir
        )
        trainer.fit()

    # 4. Evaluation & Visualization Plots
    print("Running evaluation on validation and test splits...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_weights = os.path.join(experiment_dir, "weights_best.pth")
    
    if os.path.exists(best_weights):
        checkpoint = torch.load(best_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded best weights from: {best_weights}")
    
    model = model.to(device)
    
    # Test evaluation
    y_true, y_pred, y_prob = test_model(model, test_loader, device, model_cfg.get("head", "edl"))
    results = calculate_classification_metrics(y_true, y_pred, y_prob)
    ece = calculate_ece(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob)
    
    results["ece"] = ece
    results["brier_score"] = brier
    
    # Save reports
    save_reports(results, experiment_dir, args.experiment_name)
    
    # Generate confusion matrix and reliability plots
    plot_confusion_matrix(results["confusion_matrix"], ["CNV", "DME", "DRUSEN", "NORMAL"], os.path.join(experiment_dir, "confusion_matrix.png"))
    plot_reliability_diagram(y_true, y_prob, num_bins=10, save_path=os.path.join(experiment_dir, "reliability_diagram.png"))

    # Generate LayerCAM visual explanations
    print("Generating LayerCAM visual explanations...")
    sample_scan = None
    for split in ["test", "val", "train"]:
        split_path = dataset_cfg.get("paths", {}).get(split, "")
        if os.path.exists(split_path):
            for folder in ["NORMAL", "CNV", "DME", "DRUSEN"]:
                folder_path = os.path.join(split_path, folder)
                if os.path.exists(folder_path):
                    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                    if imgs:
                        sample_scan = os.path.join(folder_path, imgs[0])
                        break
            if sample_scan:
                break

    if sample_scan and os.path.exists(sample_scan):
        # target layers for resnet50
        target_layer_gradcam = [model.backbone.layer4]
        target_layer_layercam = [model.backbone.layer3]
        compare_and_save_visualizations(
            model=model,
            target_layers_gradcam=target_layer_gradcam,
            target_layers_layercam=target_layer_layercam,
            image_path=sample_scan,
            target_class=3, # target normal
            output_dir=experiment_dir,
            prefix="explainability_check"
        )


if __name__ == "__main__":
    main()
