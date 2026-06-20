from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping untuk menghentikan training ketika performa validasi
    tidak membaik dalam beberapa epoch.

    Monitor yang umum digunakan:
    - val_loss: semakin kecil semakin baik
    - val_acc : semakin besar semakin baik
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor

        self.best_score = None
        self.counter = 0
        self.should_stop = False

        if monitor not in ["val_loss", "val_acc"]:
            raise ValueError(
                "monitor harus 'val_loss' atau 'val_acc'. "
                f"Nilai saat ini: {monitor}"
            )

    def _is_improvement(self, current_score: float) -> bool:
        if self.best_score is None:
            return True

        if self.monitor == "val_loss":
            return current_score < self.best_score - self.min_delta

        if self.monitor == "val_acc":
            return current_score > self.best_score + self.min_delta

        return False

    def step(self, current_score: float) -> bool:
        """
        Mengembalikan True jika training harus dihentikan.
        """
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop