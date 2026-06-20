from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


CLASS_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_history(history, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Validation")
    axes[0].set_title("Loss per Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Train")
    axes[1].plot(epochs, history["val_acc"], marker="o", linewidth=2, label="Validation")
    axes[1].set_title("Accuracy per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.suptitle("Classifier Training Curve", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gan_history(history, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_g_loss"]) + 1)
    has_detail_metrics = "val_detail_score" in history
    column_count = 3 if has_detail_metrics else 2
    fig, axes = plt.subplots(1, column_count, figsize=(6 * column_count, 4.8))

    axes[0].plot(epochs, history["train_g_loss"], marker="o", linewidth=2, label="Train G")
    axes[0].plot(epochs, history["val_g_loss"], marker="o", linewidth=2, label="Val G")
    axes[0].set_title("Generator Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(epochs, history["train_d_loss"], marker="o", linewidth=2, label="Train D")
    axes[1].plot(epochs, history["val_d_loss"], marker="o", linewidth=2, label="Val D")
    axes[1].set_title("Discriminator Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    if has_detail_metrics:
        axes[2].plot(
            epochs,
            history["train_detail_score"],
            linewidth=2,
            label="Train detail",
        )
        axes[2].plot(
            epochs,
            history["val_detail_score"],
            linewidth=2,
            label="Val detail",
        )
        axes[2].plot(
            epochs,
            history["val_checkpoint_score"],
            linestyle="--",
            linewidth=1.5,
            label="Checkpoint score",
        )
        axes[2].set_title("Contour / Detail Score", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Lower is better")
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].legend()

    fig.suptitle("Conditional GAN Training Curve", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    num_classes = len(class_names)

    fig_size = max(5.5, num_classes * 1.15)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    diagonal_color = "#B7D7A8"
    off_diagonal_color = "#FFFFFF"
    grid_color = "#BFBFBF"
    text_color = "#000000"

    for row in range(num_classes):
        for col in range(num_classes):
            cell_color = diagonal_color if row == col else off_diagonal_color

            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cell_color,
                    edgecolor=grid_color,
                    linewidth=1.0,
                )
            )

            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold" if row == col else "normal",
            )

    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)
    ax.set_aspect("equal")

    tick_positions = np.arange(num_classes)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_names, rotation=0, ha="center", fontsize=12, color="black")
    ax.set_yticklabels(class_names, rotation=0, fontsize=12, color="black")

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=22, color="black")
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", labelpad=10, color="black")
    ax.set_ylabel("True", fontsize=12, fontweight="bold", labelpad=10, color="black")
    ax.tick_params(axis="both", which="major", length=0, pad=7, colors="black")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_latent_space_distribution(
    real_projection,
    real_labels,
    synthetic_projection,
    synthetic_labels,
    class_names,
    explained_variance: float,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    for class_index, class_name in enumerate(class_names):
        color = CLASS_COLORS[class_index % len(CLASS_COLORS)]

        real_mask = real_labels == class_index
        syn_mask = synthetic_labels == class_index

        if np.any(real_mask):
            ax.scatter(
                real_projection[real_mask, 0],
                real_projection[real_mask, 1],
                s=28,
                alpha=0.65,
                color=color,
                marker="o",
                label=f"{class_name} - real",
            )

        if np.any(syn_mask):
            ax.scatter(
                synthetic_projection[syn_mask, 0],
                synthetic_projection[syn_mask, 1],
                s=40,
                alpha=0.85,
                color=color,
                marker="^",
                edgecolors="black",
                linewidths=0.3,
                label=f"{class_name} - synthetic",
            )

    ax.set_title("Distribusi Embedding Discriminator (PCA 2D)", fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.text(
        0.01,
        0.01,
        f"Explained variance: {explained_variance * 100:.2f}%",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper right",
        fontsize=8,
        frameon=True,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gan_feature_distribution(
    real_scores: Iterable[float],
    synthetic_scores: Iterable[float],
    synthetic_feature_distance: Iterable[float],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_scores = list(real_scores) or [0.0]
    synthetic_scores = list(synthetic_scores) or [0.0]
    synthetic_feature_distance = list(synthetic_feature_distance) or [0.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(
        real_scores,
        bins=20,
        color="#4C78A8",
        alpha=0.7,
        edgecolor="white",
        label="Real",
    )
    axes[0].hist(
        synthetic_scores,
        bins=20,
        color="#F58518",
        alpha=0.7,
        edgecolor="white",
        label="Synthetic",
    )
    axes[0].set_title("Discriminator Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Probability Real")
    axes[0].set_ylabel("Jumlah Sampel")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].boxplot(
        [synthetic_feature_distance],
        labels=["Synthetic vs\nNearest Real"],
        patch_artist=True,
        boxprops={"facecolor": "#A0CBE8"},
        medianprops={"color": "#C44E52", "linewidth": 2},
    )
    axes[1].set_title("Feature Distance pada Embedding Space", fontweight="bold")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gan_gallery(
    real_images,
    generated_images,
    labels,
    class_names,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(real_images) == 0:
        return

    rows = len(real_images)
    fig, axes = plt.subplots(rows, 2, figsize=(6.8, max(3.0, rows * 2.2)))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(rows):
        original = _tensor_to_image(real_images[row])
        generated = _tensor_to_image(generated_images[row])
        class_name = class_names[labels[row]]

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Real - {class_name}", fontsize=10, fontweight="bold")
        axes[row, 1].imshow(generated)
        axes[row, 1].set_title("GAN Synthetic", fontsize=10, fontweight="bold")

        for col in range(2):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return array
