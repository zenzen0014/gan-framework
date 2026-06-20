import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision.utils import save_image


def denormalize_gan_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def encode_loader(model, data_loader, device):
    model.eval()

    latent_batches = []
    label_batches = []

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[:2]
            images = images.to(device)
            labels = labels.to(device)

            latents = model.encode(images, labels)
            latent_batches.append(latents.cpu())
            label_batches.append(labels.cpu())

    if not latent_batches:
        feature_dim = getattr(model, "feature_dim", getattr(model, "latent_dim", 0))
        return np.empty((0, feature_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)

    latents = torch.cat(latent_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()
    return latents, labels


def compute_latent_statistics(latents, labels, class_names) -> Dict[str, Dict[str, np.ndarray | float | int]]:
    stats = {}

    for class_index, class_name in enumerate(class_names):
        class_latents = latents[labels == class_index]
        if len(class_latents) == 0:
            continue

        mean = class_latents.mean(axis=0)
        std = class_latents.std(axis=0)
        distances = np.linalg.norm(class_latents - mean, axis=1)

        stats[class_name] = {
            "mean": mean,
            "std": std,
            "sample_count": int(len(class_latents)),
            "avg_radius": float(distances.mean()),
            "max_radius": float(distances.max()),
        }

    return stats


def save_latent_statistics_json(stats, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for class_name, values in stats.items():
        serializable[class_name] = {
            "sample_count": values["sample_count"],
            "avg_radius": values["avg_radius"],
            "max_radius": values["max_radius"],
            "mean_preview": np.asarray(values["mean"])[:8].round(6).tolist(),
            "std_preview": np.asarray(values["std"])[:8].round(6).tolist(),
        }

    output_path.write_text(
        json.dumps(serializable, indent=4),
        encoding="utf-8",
    )


def generate_synthetic_images(
    model,
    class_names,
    output_dir: str | Path,
    images_per_class: int,
    noise_scale: float,
    device,
    seed: int,
    batch_size: int = 32,
) -> List[Tuple[Path, int]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    saved_samples: List[Tuple[Path, int]] = []
    model.eval()

    with torch.no_grad():
        for class_index, class_name in enumerate(class_names):
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for old_file in class_dir.iterdir():
                if old_file.is_file() and old_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    old_file.unlink()

            total_saved = 0
            while total_saved < images_per_class:
                current_batch = min(batch_size, images_per_class - total_saved)
                class_labels = torch.full(
                    (current_batch,),
                    fill_value=class_index,
                    device=device,
                    dtype=torch.long,
                )

                generated = model.generate(
                    labels=class_labels,
                    noise_scale=noise_scale,
                ).cpu()
                generated = denormalize_gan_tensor(generated)

                for batch_index in range(current_batch):
                    image_path = class_dir / f"{class_name}_synthetic_{total_saved + batch_index + 1:04d}.png"
                    save_image(generated[batch_index], image_path)
                    saved_samples.append((image_path, class_index))

                total_saved += current_batch

    return saved_samples


def sample_for_visualization(latents, labels, max_points: int, seed: int):
    if len(latents) <= max_points:
        return latents, labels

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(latents), size=max_points, replace=False)
    return latents[indices], labels[indices]


def project_latent_space(
    real_latents,
    real_labels,
    synthetic_latents,
    synthetic_labels,
    max_points: int,
    seed: int,
):
    if len(real_latents) == 0 or len(synthetic_latents) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            0.0,
        )

    real_latents, real_labels = sample_for_visualization(
        real_latents,
        real_labels,
        max_points=max_points,
        seed=seed,
    )
    synthetic_latents, synthetic_labels = sample_for_visualization(
        synthetic_latents,
        synthetic_labels,
        max_points=max_points,
        seed=seed + 1,
    )

    combined = np.vstack([real_latents, synthetic_latents])
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(combined)

    split_index = len(real_latents)
    return (
        projected[:split_index],
        real_labels,
        projected[split_index:],
        synthetic_labels,
        float(pca.explained_variance_ratio_.sum()),
    )


def build_latent_space_metrics(
    real_latents,
    real_labels,
    synthetic_latents,
    synthetic_labels,
    class_names,
):
    metrics = {
        "real_sample_count": int(len(real_latents)),
        "synthetic_sample_count": int(len(synthetic_latents)),
        "per_class": {},
    }

    for class_index, class_name in enumerate(class_names):
        real_class = real_latents[real_labels == class_index]
        synthetic_class = synthetic_latents[synthetic_labels == class_index]

        if len(real_class) == 0 or len(synthetic_class) == 0:
            continue

        real_centroid = real_class.mean(axis=0)
        synthetic_centroid = synthetic_class.mean(axis=0)
        centroid_shift = np.linalg.norm(real_centroid - synthetic_centroid)
        real_radius = np.linalg.norm(real_class - real_centroid, axis=1).mean()
        synthetic_radius = np.linalg.norm(synthetic_class - synthetic_centroid, axis=1).mean()

        metrics["per_class"][class_name] = {
            "real_count": int(len(real_class)),
            "synthetic_count": int(len(synthetic_class)),
            "centroid_shift": float(centroid_shift),
            "real_avg_radius": float(real_radius),
            "synthetic_avg_radius": float(synthetic_radius),
        }

    return metrics


def collect_gan_analysis_batches(
    model,
    data_loader,
    device,
    preview_count: int,
    noise_scale: float,
    config=None,
):
    model.eval()

    preview_real_images = []
    preview_fake_images = []
    preview_labels = []
    real_scores = []
    synthetic_scores = []
    real_latents_by_class: Dict[int, List[np.ndarray]] = {}

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            real_logits, _, real_latents = model.discriminate(images, labels)
            fake_images = model.generate(labels=labels, noise_scale=noise_scale)
            fake_logits, _, _ = model.discriminate(fake_images, labels)

            if getattr(config, "gan_loss_mode", "bce").lower() == "wgan_gp":
                real_scores.extend(real_logits.cpu().tolist())
                synthetic_scores.extend(fake_logits.cpu().tolist())
            else:
                real_scores.extend(torch.sigmoid(real_logits).cpu().tolist())
                synthetic_scores.extend(torch.sigmoid(fake_logits).cpu().tolist())

            for class_index in labels.unique().tolist():
                class_mask = labels == class_index
                class_latents = real_latents[class_mask].cpu().numpy()
                real_latents_by_class.setdefault(int(class_index), []).extend(class_latents)

            if len(preview_real_images) < preview_count:
                remaining = preview_count - len(preview_real_images)
                preview_real_images.extend(
                    denormalize_gan_tensor(images[:remaining].cpu())
                )
                preview_fake_images.extend(
                    denormalize_gan_tensor(fake_images[:remaining].cpu())
                )
                preview_labels.extend(labels[:remaining].cpu().tolist())

    for class_index, latents in real_latents_by_class.items():
        real_latents_by_class[class_index] = np.asarray(latents, dtype=np.float32)

    return {
        "preview_real_images": preview_real_images,
        "preview_fake_images": preview_fake_images,
        "preview_labels": preview_labels,
        "real_scores": real_scores,
        "synthetic_scores": synthetic_scores,
        "real_latents_by_class": real_latents_by_class,
    }


def collect_synthetic_feature_distances(
    model,
    data_loader,
    real_latents_by_class,
    device,
) -> List[float]:
    model.eval()
    distances: List[float] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            synthetic_latents = model.encode(images, labels).cpu().numpy()
            label_values = labels.cpu().numpy()

            for latent, label in zip(synthetic_latents, label_values):
                real_latents = real_latents_by_class.get(int(label))
                if real_latents is None or len(real_latents) == 0:
                    continue

                class_distances = np.linalg.norm(real_latents - latent, axis=1)
                distances.append(float(class_distances.min()))

    return distances


def summarize_feature_metrics(
    real_scores,
    synthetic_scores,
    synthetic_feature_distance,
    class_stats,
    class_names,
):
    def safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0

    def safe_std(values):
        return float(np.std(values)) if len(values) > 0 else 0.0

    per_class_summary = {}
    for class_name in class_names:
        if class_name not in class_stats:
            continue
        stats = class_stats[class_name]
        per_class_summary[class_name] = {
            "sample_count": int(
                stats.get(
                    "sample_count",
                    stats.get("real_count", 0),
                )
            ),
            "real_avg_radius": float(
                stats.get(
                    "avg_radius",
                    stats.get("real_avg_radius", 0.0),
                )
            ),
            "real_max_radius": float(stats.get("max_radius", 0.0)),
        }

    return {
        "real_score_mean": safe_mean(real_scores),
        "real_score_std": safe_std(real_scores),
        "synthetic_score_mean": safe_mean(synthetic_scores),
        "synthetic_score_std": safe_std(synthetic_scores),
        "synthetic_feature_distance_mean": safe_mean(synthetic_feature_distance),
        "synthetic_feature_distance_std": safe_std(synthetic_feature_distance),
        "per_class": per_class_summary,
    }


def save_feature_metrics_json(metrics, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
