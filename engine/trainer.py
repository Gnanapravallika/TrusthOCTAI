"""Trainer loop module for training TrustOCT models."""

import os
import sys
import time
from typing import Dict, Optional, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from engine.losses import EdlLoss
from models.trustoct import coral_loss


class Trainer:
    """Rigorous training and validation engine for TrustOCT models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_config_path: str,
        model_config_path: str,
        experiment_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir

        self.train_cfg = self._load_yaml(train_config_path)
        self.model_cfg = self._load_yaml(model_config_path)

        # Set device
        device_str = self.train_cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
        self.model = self.model.to(self.device)

        # Build optimizer
        opt_type = self.train_cfg.get("optimizer", "adamw").lower()
        lr = self.train_cfg.get("learning_rate", 1e-4)
        wd = self.train_cfg.get("weight_decay", 1e-4)
        
        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        # Build scheduler
        self.epochs = self.train_cfg.get("epochs", 30)
        sched_type = self.train_cfg.get("scheduler", "cosine").lower()
        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_cfg.get("cosine_T_max", self.epochs)
            )
        else:
            self.scheduler = None

        # Build primary loss function
        self.head_type = self.model_cfg.get("head", "edl").lower()
        self.num_classes = self.model_cfg.get("num_classes", 4)
        
        if self.head_type == "edl":
            edl_cfg = self.train_cfg.get("edl", {})
            self.criterion = EdlLoss(
                num_classes=self.num_classes,
                annealing_epochs=edl_cfg.get("annealing_epochs", 10)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None

        log_cfg = self.train_cfg.get("logging", {})
        self.tb_writer = None
        if log_cfg.get("tensorboard", True):
            tb_dir = os.path.join(experiment_dir, "tb_logs")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = self.train_cfg.get("checkpoint", {}).get("patience", 7)

    def _load_yaml(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if self.head_type == "edl":
                        _, alpha = outputs
                        loss = self.criterion(alpha, targets, epoch)
                        preds = torch.argmax(alpha, dim=1)
                    else:
                        loss = self.criterion(outputs, targets)
                        preds = torch.argmax(outputs, dim=1)

                    if self.model_cfg.get("domain_generalization", "identity").lower() == "coral":
                        if images.size(0) >= 4:
                            half = images.size(0) // 2
                            last_feats = self.model.dg.last_features
                            c_loss = coral_loss(last_feats[:half], last_feats[half:])
                            loss += self.train_cfg.get("coral", {}).get("weight", 0.5) * c_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if self.head_type == "edl":
                    _, alpha = outputs
                    loss = self.criterion(alpha, targets, epoch)
                    preds = torch.argmax(alpha, dim=1)
                else:
                    loss = self.criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                if self.head_type == "edl":
                    _, alpha = outputs
                    loss = self.criterion(alpha, targets, epoch)
                    preds = torch.argmax(alpha, dim=1)
                else:
                    loss = self.criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                running_loss += loss.item() * images.size(0)
                correct += (preds == targets).sum().item()
                total += images.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self) -> None:
        print(f"Starting training on device: {self.device}")
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr:.6f}")

            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", val_loss, epoch)
                self.tb_writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.tb_writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.tb_writer.add_scalar("LR", lr, epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1} due to validation loss plateau.")
                    break

            if (epoch + 1) % self.train_cfg.get("checkpoint", {}).get("save_freq", 5) == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)

        if self.tb_writer:
            self.tb_writer.close()
        print("Training complete.")

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        os.makedirs(self.experiment_dir, exist_ok=True)
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_val_loss,
            "val_loss": val_loss
        }

        if is_best:
            filepath = os.path.join(self.experiment_dir, "weights_best.pth")
        else:
            filepath = os.path.join(self.experiment_dir, f"checkpoint_epoch_{epoch+1}.pth")

        torch.save(state, filepath)
        print(f"Saved weights: {os.path.basename(filepath)}")


def run_experiment(
    model_name: str,
    train_df,
    val_df,
    test_df,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    device_str: str = "cuda"
):
    """Run the entire training, validation, and evaluation pipeline for a specific model configuration."""
    import os
    import pandas as pd
    from torch.utils.data import DataLoader
    from datasets.dataset import DataFrameOCTDataset
    from datasets.transforms import get_train_transforms, get_val_transforms
    from models.trustoct import TrustOCT
    from engine.tester import test_model
    from evaluation.metrics import calculate_classification_metrics, plot_confusion_matrix
    from evaluation.calibration import calculate_ece, calculate_brier_score, plot_reliability_diagram
    from evaluation.report import save_reports
    
    # 1. Determine model settings
    model_name_lower = model_name.lower()
    if model_name_lower == "resnet50":
        use_multiscale = False
        use_cbam = False
    elif model_name_lower in ["resnet50_msf", "msf_resnet50"]:
        use_multiscale = True
        use_cbam = False
    elif model_name_lower in ["msf_cbam_resnet50", "trustoct"]:
        use_multiscale = True
        use_cbam = True
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    print(f"\n==============================================")
    print(f"Starting Experiment: {model_name}")
    print(f"  - MSF Fusion: {use_multiscale}")
    print(f"  - CBAM Gate:  {use_cbam}")
    print(f"==============================================")

    # 2. Build datasets & load transforms
    combined_cfg = {
        "preprocessing": {
            "resize": [224, 224],
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
            "bilateral_filter": {"enabled": True, "d": 9, "sigma_color": 75.0, "sigma_space": 75.0}
        },
        "augmentations": {
            "random_rotate": {"limit": 15, "p": 0.5},
            "horizontal_flip": {"p": 0.5},
            "random_brightness_contrast": {"brightness_limit": 0.1, "contrast_limit": 0.1, "p": 0.5},
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        }
    }
    
    train_transform = get_train_transforms(combined_cfg)
    val_transform = get_val_transforms(combined_cfg)

    train_dataset = DataFrameOCTDataset(train_df, transform=train_transform)
    val_dataset = DataFrameOCTDataset(val_df, transform=val_transform)
    test_dataset = DataFrameOCTDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Compile model
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = TrustOCT(
        backbone_name="resnet50",
        pretrained=True,
        use_multiscale=use_multiscale,
        use_cbam=use_cbam,
        dg_type="identity",
        head_type="softmax",
        num_classes=4,
        dropout_prob=0.5
    ).to(device)

    # 4. Compile Trainer configurations
    experiment_dir = f"outputs/{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save temporary yaml files for the trainer initializer requirements
    temp_train_yaml = os.path.join(experiment_dir, "train_temp.yaml")
    temp_model_yaml = os.path.join(experiment_dir, "model_temp.yaml")
    
    train_yaml_data = {
        "device": device_str,
        "epochs": epochs,
        "optimizer": "adamw",
        "learning_rate": lr,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "checkpoint": {"patience": 7, "save_freq": 5},
        "logging": {"tensorboard": False}
    }
    model_yaml_data = {
        "head": "softmax",
        "num_classes": 4,
        "domain_generalization": "identity"
    }
    
    with open(temp_train_yaml, "w") as f:
        yaml.dump(train_yaml_data, f)
    with open(temp_model_yaml, "w") as f:
        yaml.dump(model_yaml_data, f)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_config_path=temp_train_yaml,
        model_config_path=temp_model_yaml,
        experiment_dir=experiment_dir
    )

    # 5. Fit
    trainer.fit()

    # 6. Evaluate on test set
    best_weights = os.path.join(experiment_dir, "weights_best.pth")
    if os.path.exists(best_weights):
        checkpoint = torch.load(best_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded best weights from {best_weights}")
        
    y_true, y_pred, y_prob = test_model(model, test_loader, device, "softmax")
    results = calculate_classification_metrics(y_true, y_pred, y_prob)
    ece = calculate_ece(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob)
    results["ece"] = ece
    results["brier_score"] = brier
    
    save_reports(results, experiment_dir, model_name)
    plot_confusion_matrix(results["confusion_matrix"], ["CNV", "DME", "DRUSEN", "NORMAL"], os.path.join(experiment_dir, "confusion_matrix.png"))
    plot_reliability_diagram(y_true, y_prob, num_bins=10, save_path=os.path.join(experiment_dir, "reliability_diagram.png"))

    print(f"==============================================")
    print(f"Completed Experiment: {model_name}")
    print(f"Accuracy:    {results['accuracy']*100:.2f}%")
    print(f"Macro F1:    {results['f1_score']*100:.2f}%")
    print(f"ECE Score:   {ece:.4f}")
    print(f"==============================================")
    
    return results

