from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.early_stopping import EarlyStopping


def run_autoencoder_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    is_train: bool,
) -> float:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            reconstructions, _ = model(images, labels)
            loss = criterion(reconstructions, images)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def save_autoencoder_checkpoint(
    path: str | Path,
    model,
    class_names,
    config,
    best_val_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "config": asdict(config),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    save_path: str | Path,
    class_names,
    config,
) -> Tuple[Dict[str, list], float]:
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")

    early_stopping = None
    if config.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            monitor="val_loss",
        )

    print("Autoencoder dilatih maksimal sebanyak:", epochs, "epochs")
    print("Checkpoint autoencoder disimpan di :", save_path)
    if early_stopping is not None:
        print("Early stopping AE     : aktif (monitor=val_loss)")
    else:
        print("Early stopping AE     : tidak aktif")

    print()
    print(
        f"{'Epoch':>5} | "
        f"{'Train Loss':>10} | "
        f"{'Val Loss':>8} | "
        f"{'Early Stop':>12}"
    )
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        train_loss = run_autoencoder_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
        )
        val_loss = run_autoencoder_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_autoencoder_checkpoint(
                path=save_path,
                model=model,
                class_names=class_names,
                config=config,
                best_val_loss=best_val_loss,
            )

        early_stop_text = "-"
        if early_stopping is not None:
            should_stop = early_stopping.step(val_loss)
            early_stop_text = f"{early_stopping.counter}/{early_stopping.patience}"
            if should_stop:
                early_stop_text = "STOP"

        marker = " *best" if is_best else ""
        print(
            f"{epoch:5d} | "
            f"{train_loss:10.4f} | "
            f"{val_loss:8.4f} | "
            f"{early_stop_text:>12}"
            f"{marker}"
        )

        if early_stopping is not None and early_stopping.should_stop:
            print()
            print(
                f"Early stopping AE aktif pada epoch {epoch}. "
                "Training dihentikan karena val_loss tidak membaik."
            )
            break

    return history, best_val_loss


def load_autoencoder_checkpoint(model, checkpoint_path: str | Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


class CombinedReconstructionLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 0.70,
        mse_weight: float = 0.20,
        edge_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.edge_weight = edge_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        if self.l1_weight > 0:
            total_loss = total_loss + self.l1_weight * self.l1_loss(prediction, target)

        if self.mse_weight > 0:
            total_loss = total_loss + self.mse_weight * self.mse_loss(prediction, target)

        if self.edge_weight > 0:
            total_loss = total_loss + self.edge_weight * self.edge_loss(prediction, target)

        return total_loss

    @staticmethod
    def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        prediction_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = F.l1_loss(prediction_dx, target_dx)
        loss_y = F.l1_loss(prediction_dy, target_dy)
        return 0.5 * (loss_x + loss_y)


def build_reconstruction_loss(config) -> nn.Module:
    return CombinedReconstructionLoss(
        l1_weight=config.ae_loss_l1_weight,
        mse_weight=config.ae_loss_mse_weight,
        edge_weight=config.ae_loss_edge_weight,
    )
