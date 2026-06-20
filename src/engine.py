from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch

from src.early_stopping import EarlyStopping


def run_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    is_train: bool,
) -> Tuple[float, float]:
    """
    Menjalankan satu epoch untuk training atau validation.

    Return:
    - average_loss
    - accuracy dalam persen
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def save_checkpoint(
    path: str | Path,
    model,
    class_names,
    config,
    best_val_accuracy: float,
) -> None:
    """
    Menyimpan model terbaik beserta metadata eksperimen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "config": asdict(config),
            "best_val_accuracy": best_val_accuracy,
        },
        path,
    )


def create_early_stopping(config):
    """
    Membuat object EarlyStopping jika fitur early stopping diaktifkan.

    Jika EARLY_STOPPING=false, maka fungsi ini mengembalikan None.
    """
    if not config.early_stopping:
        return None

    return EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        monitor=config.early_stopping_monitor,
    )


def get_monitor_value(
    monitor_name: str,
    val_loss: float,
    val_acc: float,
) -> float:
    """
    Mengambil nilai yang akan dipantau oleh early stopping.

    Pilihan:
    - val_loss: semakin kecil semakin baik
    - val_acc : semakin besar semakin baik
    """
    if monitor_name == "val_loss":
        return val_loss

    if monitor_name == "val_acc":
        return val_acc

    raise ValueError(
        "EARLY_STOPPING_MONITOR harus 'val_loss' atau 'val_acc'. "
        f"Nilai saat ini: {monitor_name}"
    )


def train_model(
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
    """
    Training model sampai epoch selesai atau sampai early stopping aktif.

    Model terbaik tetap disimpan berdasarkan validation accuracy tertinggi.
    Early stopping digunakan untuk menghentikan training jika validation metric
    tidak membaik dalam beberapa epoch.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_accuracy = 0.0
    early_stopping = create_early_stopping(config)

    print("Model dilatih maksimal sebanyak:", epochs, "epochs")
    print("Model terbaik akan disimpan di:", save_path)

    if early_stopping is not None:
        print("Early stopping       : aktif")
        print("Monitor              :", config.early_stopping_monitor)
        print("Patience             :", config.early_stopping_patience)
        print("Minimum delta        :", config.early_stopping_min_delta)
    else:
        print("Early stopping       : tidak aktif")

    print()
    print(
        f"{'Epoch':>5} | "
        f"{'Train Loss':>10} | "
        f"{'Train Acc':>9} | "
        f"{'Val Loss':>8} | "
        f"{'Val Acc':>7} | "
        f"{'Early Stop':>12}"
    )
    print("-" * 82)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_accuracy

        if is_best:
            best_val_accuracy = val_acc

            save_checkpoint(
                path=save_path,
                model=model,
                class_names=class_names,
                config=config,
                best_val_accuracy=best_val_accuracy,
            )

        early_stop_text = "-"

        if early_stopping is not None:
            monitor_value = get_monitor_value(
                monitor_name=config.early_stopping_monitor,
                val_loss=val_loss,
                val_acc=val_acc,
            )

            should_stop = early_stopping.step(monitor_value)

            early_stop_text = (
                f"{early_stopping.counter}/"
                f"{early_stopping.patience}"
            )

            if should_stop:
                early_stop_text = "STOP"

        marker = " *best" if is_best else ""

        print(
            f"{epoch:5d} | "
            f"{train_loss:10.4f} | "
            f"{train_acc:8.2f}% | "
            f"{val_loss:8.4f} | "
            f"{val_acc:6.2f}% | "
            f"{early_stop_text:>12}"
            f"{marker}"
        )

        if early_stopping is not None and early_stopping.should_stop:
            print()
            print(
                f"Early stopping aktif pada epoch {epoch}. "
                f"Training dihentikan karena "
                f"{config.early_stopping_monitor} tidak membaik selama "
                f"{config.early_stopping_patience} epoch."
            )
            break

    return history, best_val_accuracy