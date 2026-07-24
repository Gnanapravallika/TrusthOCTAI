"""Training engine for TrustOCT models."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine.losses import get_loss_function


class TrustOCTTrainer:
    """Trainer class for model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: torch.device = None,
        epochs: int = 30,
        head_type: str = "softmax",
        experiment_dir: str = "outputs/resnet50",
        patience: int = 7
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.head_type = head_type
        self.experiment_dir = experiment_dir
        self.patience = patience

        self.criterion = get_loss_function("edl" if head_type == "edl" else "ce")
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == "cuda" else None

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.patience_counter = 0

        os.makedirs(self.experiment_dir, exist_ok=True)

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.train_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    if self.head_type == "edl":
                        evidence, alpha = outputs
                        loss = self.criterion(alpha, targets, epoch)
                        preds = torch.argmax(alpha, dim=1)
                    else:
                        loss = self.criterion(outputs, targets)
                        preds = torch.argmax(outputs, dim=1)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if self.head_type == "edl":
                    evidence, alpha = outputs
                    loss = self.criterion(alpha, targets, epoch)
                    preds = torch.argmax(alpha, dim=1)
                else:
                    loss = self.criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.val_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            if self.head_type == "edl":
                evidence, alpha = self.model(images)
                loss = self.criterion(alpha, targets, epoch)
                preds = torch.argmax(alpha, dim=1)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self):
        exp_name = os.path.basename(self.experiment_dir)
        print(f"Training {exp_name} for {self.epochs} epochs on {self.device}...", flush=True)

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}",
                flush=True
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
                print(f"✅ Best model updated! Val Acc: {val_acc:.4f}", flush=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
                    break

        print(f"Training complete. Best Val Acc: {self.best_val_acc:.4f}", flush=True)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": self.best_val_acc
        }
        if is_best:
            path = os.path.join(self.experiment_dir, "weights_best.pth")
            torch.save(state, path)


def run_experiment(model_name: str, train_df, val_df, test_df, epochs: int = 30, lr: float = 1e-4, batch_size: int = 32):
    """Helper runner function to train models for ablation experiments."""
    from oct_datasets.dataset import DataFrameOCTDataset
    from oct_datasets.transforms import get_train_transforms, get_val_transforms
    from models.trustoct import TrustOCT

    train_trans = get_train_transforms()
    val_trans = get_val_transforms()

    train_ds = DataFrameOCTDataset(train_df, transform=train_trans)
    val_ds = DataFrameOCTDataset(val_df, transform=val_trans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    use_msf = True if "msf" in model_name else False
    use_cbam = True if "cbam" in model_name else False

    print("\n" + "═" * 60)
    print(f"  🔬  EXPERIMENT  —  {model_name.upper()}")
    print("═" * 60)
    print(f"      Multi-Scale Fusion (MSF) : {'✅' if use_msf else '❌'}")
    print(f"      CBAM Attention Gate      : {'✅' if use_cbam else '❌'}")
    print("─" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrustOCT(
        backbone_name="resnet50",
        pretrained=True,
        use_multiscale=use_msf,
        use_cbam=use_cbam,
        head_type="softmax",
        num_classes=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    exp_dir = f"outputs/{model_name}"
    trainer = TrustOCTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        head_type="softmax",
        experiment_dir=exp_dir
    )

    trainer.fit()
    return exp_dir
