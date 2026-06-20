from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_checkpoint(model, checkpoint_path: str | Path, device):
    """Memuat bobot model terbaik."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def collect_predictions(model, data_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Mengumpulkan label asli dan prediksi model."""
    model.eval()

    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_predictions)


def make_classification_report(
    labels,
    predictions,
    class_names,
) -> str:
    """Membuat classification report dalam bentuk teks."""
    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )


def calculate_accuracy(labels, predictions) -> float:
    """Menghitung akurasi dalam persen."""
    return 100.0 * accuracy_score(labels, predictions)


def make_confusion_matrix(labels, predictions):
    """Membuat confusion matrix."""
    return confusion_matrix(labels, predictions)
