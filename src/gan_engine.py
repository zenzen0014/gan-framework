import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from src.early_stopping import EarlyStopping


def save_gan_checkpoint(path: str | Path, model, class_names, config, best_val_generator_loss: float, best_val_selection_score: float | None = None, epoch: int | None = None, val_metrics: Dict[str, float] | None = None, generator_state_dict: Dict[str, torch.Tensor] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_state_dict": generator_state_dict if generator_state_dict is not None else model.generator.state_dict(),
            "discriminator_state_dict": model.discriminator.state_dict(),
            "class_names": class_names,
            "config": asdict(config),
            "best_val_generator_loss": best_val_generator_loss,
            "best_val_selection_score": best_val_selection_score,
            "best_epoch": epoch,
            "best_val_metrics": val_metrics,
        },
        path,
    )


def load_gan_checkpoint(model, checkpoint_path: str | Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(checkpoint["generator_state_dict"])
    model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    return checkpoint


def _hinge_loss_dis(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))


def _hinge_loss_gen(fake_logits: torch.Tensor) -> torch.Tensor:
    return -torch.mean(fake_logits)


def _r1_penalty(model, real_images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    real_images = real_images.detach().clone().requires_grad_(True)
    real_logits, _, _ = model.discriminate(real_images, labels)
    gradients = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradients.pow(2).reshape(gradients.size(0), -1).sum(dim=1).mean()


def _feature_maps(model, images: torch.Tensor, labels: torch.Tensor):
    if hasattr(model.discriminator, "forward_with_maps"):
        return model.discriminator.forward_with_maps(images, labels)
    logits, _, features = model.discriminate(images, labels)
    return logits, [features.view(features.size(0), features.size(1), 1, 1)]


def _compute_classwise_feature_match_loss(fake_maps, real_maps, labels: torch.Tensor) -> torch.Tensor:
    real_last = real_maps[-1].mean(dim=(2, 3))
    fake_last = fake_maps[-1].mean(dim=(2, 3))
    losses = []
    for class_id in labels.unique(sorted=True):
        mask = labels == class_id
        if mask.sum().item() == 0:
            continue
        losses.append(F.mse_loss(fake_last[mask].mean(dim=0), real_last[mask].mean(dim=0)))
    if not losses:
        return fake_last.new_tensor(0.0)
    return torch.stack(losses).mean()


def _checkpoint_selection_score(metrics: Dict[str, float], config) -> float:
    return metrics["generator_loss"]


def _denormalize_gan_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


def _make_preview_labels(class_names, count: int, device) -> torch.Tensor:
    num_classes = max(1, len(class_names))
    labels = torch.arange(num_classes, device=device, dtype=torch.long)
    repeats = (count + num_classes - 1) // num_classes
    return labels.repeat(repeats)[:count]


def _save_gan_preview(model, labels: torch.Tensor, fixed_noise: torch.Tensor, output_path: str | Path, generator_override=None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = generator_override(fixed_noise, labels) if generator_override is not None else model.generate(labels=labels, noise=fixed_noise)
        generated = _denormalize_gan_tensor(generated.detach().cpu())
        save_image(generated, output_path, nrow=max(1, int(labels.numel() ** 0.5)), padding=2)
    if was_training:
        model.train()


@torch.no_grad()
def _update_ema_generator(ema_generator, generator, decay: float) -> None:
    ema_state = ema_generator.state_dict()
    generator_state = generator.state_dict()
    for name, value in generator_state.items():
        if value.dtype.is_floating_point:
            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value)


def _run_generator_step(model, real_images, labels, source_criterion, class_criterion, config):
    noise = model.sample_noise(real_images.size(0), real_images.device, config.gan_noise_scale)
    fake_images = model.generate(labels=labels, noise=noise)
    fake_logits, fake_maps = _feature_maps(model, fake_images, labels)
    with torch.no_grad():
        _, real_maps = _feature_maps(model, real_images, labels)
    adversarial_loss = _hinge_loss_gen(fake_logits)
    feature_match_loss = _compute_classwise_feature_match_loss(fake_maps, real_maps, labels)
    generator_loss = adversarial_loss + config.gan_feature_match_weight * feature_match_loss
    detail_metrics = {
        "edge_match_loss": 0.0,
        "contrast_gap": 0.0,
        "diversity_penalty": 0.0,
        "detail_score": float(generator_loss.detach().item()),
    }
    return generator_loss, fake_images, fake_logits, 1.0, float(feature_match_loss.detach().item()), detail_metrics


def _run_discriminator_step(model, real_images, labels, source_criterion, class_criterion, config, global_step: int = 0, apply_r1: bool = True):
    real_logits, _, _ = model.discriminate(real_images, labels)
    noise = model.sample_noise(real_images.size(0), real_images.device, config.gan_noise_scale)
    fake_images = model.generate(labels=labels, noise=noise).detach()
    fake_logits, _, _ = model.discriminate(fake_images, labels)
    discriminator_loss = _hinge_loss_dis(real_logits, fake_logits)
    r1_every = getattr(config, "gan_r1_every", 16)
    if apply_r1 and r1_every > 0 and global_step % r1_every == 0:
        discriminator_loss = discriminator_loss + (getattr(config, "gan_r1_gamma", 5.0) / 2.0) * _r1_penalty(model, real_images, labels)
    return discriminator_loss, real_logits, fake_logits, 1.0, 1.0


def run_gan_epoch(model, data_loader, generator_optimizer, discriminator_optimizer, source_criterion, class_criterion, device, config, is_train: bool, ema_generator=None, global_step: int = 0):
    model.train() if is_train else model.eval()
    totals = {"generator_loss": 0.0, "discriminator_loss": 0.0, "real_score": 0.0, "fake_score": 0.0, "critic_gap": 0.0, "real_class_accuracy": 0.0, "fake_class_accuracy": 0.0, "feature_match_loss": 0.0, "edge_match_loss": 0.0, "contrast_gap": 0.0, "diversity_penalty": 0.0, "detail_score": 0.0}
    total_samples = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)
            if is_train:
                for _ in range(config.gan_disc_steps):
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    discriminator_loss, real_logits, fake_logits, real_class_accuracy, fake_class_accuracy = _run_discriminator_step(model, images, labels, source_criterion, class_criterion, config, global_step, apply_r1=True)
                    discriminator_loss.backward()
                    discriminator_optimizer.step()
                    global_step += 1
                for parameter in model.discriminator.parameters():
                    parameter.requires_grad_(False)
                generator_optimizer.zero_grad(set_to_none=True)
                generator_loss, _, generator_fake_logits, generator_fake_class_accuracy, feature_match_loss, detail_metrics = _run_generator_step(model, images, labels, source_criterion, class_criterion, config)
                generator_loss.backward()
                generator_optimizer.step()
                for parameter in model.discriminator.parameters():
                    parameter.requires_grad_(True)
                if ema_generator is not None:
                    _update_ema_generator(ema_generator, model.generator, config.gan_ema_decay)
                fake_logits_for_score = generator_fake_logits
                fake_class_accuracy_for_score = generator_fake_class_accuracy
            else:
                discriminator_loss, real_logits, fake_logits, real_class_accuracy, fake_class_accuracy = _run_discriminator_step(model, images, labels, source_criterion, class_criterion, config, global_step, apply_r1=False)
                generator_loss, _, fake_logits_for_score, fake_class_accuracy_for_score, feature_match_loss, detail_metrics = _run_generator_step(model, images, labels, source_criterion, class_criterion, config)
            totals["generator_loss"] += generator_loss.item() * batch_size
            totals["discriminator_loss"] += discriminator_loss.item() * batch_size
            totals["real_score"] += real_logits.sum().item()
            totals["fake_score"] += fake_logits_for_score.sum().item()
            totals["critic_gap"] += (real_logits.sum().item() - fake_logits_for_score.sum().item())
            totals["real_class_accuracy"] += real_class_accuracy * batch_size
            totals["fake_class_accuracy"] += fake_class_accuracy_for_score * batch_size
            totals["feature_match_loss"] += feature_match_loss * batch_size
            for key in ("edge_match_loss", "contrast_gap", "diversity_penalty", "detail_score"):
                totals[key] += detail_metrics[key] * batch_size
            total_samples += batch_size
    if total_samples == 0:
        return {key: 0.0 for key in totals}, global_step
    return {key: value / total_samples for key, value in totals.items()}, global_step


def train_gan(model, train_loader, val_loader, generator_optimizer, discriminator_optimizer, source_criterion, class_criterion, device, epochs: int, save_path: str | Path, class_names, config) -> Tuple[Dict[str, list], float]:
    history = {key: [] for key in ["train_g_loss", "train_d_loss", "val_g_loss", "val_d_loss", "train_real_score", "train_fake_score", "val_real_score", "val_fake_score", "train_critic_gap", "val_critic_gap", "train_real_class_acc", "train_fake_class_acc", "val_real_class_acc", "val_fake_class_acc", "val_checkpoint_score", "train_edge_match_loss", "val_edge_match_loss", "train_contrast_gap", "val_contrast_gap", "train_diversity_penalty", "val_diversity_penalty", "train_detail_score", "val_detail_score"]}
    best_val_generator_loss = float("inf")
    best_val_selection_score = float("inf")
    early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta, "val_loss") if config.gan_early_stopping else None
    print("GAN dilatih maksimal sebanyak:", epochs, "epochs")
    print("Checkpoint GAN disimpan di   :", save_path)
    print("Loss GAN                     : hinge + R1 + classwise feature matching")
    print("Early stopping GAN          : aktif" if early_stopping is not None else "Early stopping GAN          : tidak aktif")
    print()
    print(f"{'Epoch':>5} | {'Train G':>9} | {'Train D':>9} | {'Val G':>9} | {'Val D':>9} | {'Crit R':>6} | {'Crit F':>6} | {'Gap':>6} | {'FM':>7} | {'Early Stop':>12}")
    print("-" * 100)
    preview_dir = Path(save_path).parent / "sample_previews"
    preview_count = min(config.gan_preview_grid_count, max(1, len(class_names) * 4))
    preview_labels = _make_preview_labels(class_names, preview_count, device)
    fixed_preview_noise = model.sample_noise(preview_labels.numel(), device, config.gan_noise_scale)
    ema_generator = copy.deepcopy(model.generator).to(device).eval()
    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)
    global_step = 0
    for epoch in range(1, epochs + 1):
        train_metrics, global_step = run_gan_epoch(model, train_loader, generator_optimizer, discriminator_optimizer, source_criterion, class_criterion, device, config, True, ema_generator, global_step)
        training_generator = model.generator
        model.generator = ema_generator
        val_metrics, global_step = run_gan_epoch(model, val_loader, generator_optimizer, discriminator_optimizer, source_criterion, class_criterion, device, config, False, None, global_step)
        model.generator = training_generator
        history["train_g_loss"].append(train_metrics["generator_loss"])
        history["train_d_loss"].append(train_metrics["discriminator_loss"])
        history["val_g_loss"].append(val_metrics["generator_loss"])
        history["val_d_loss"].append(val_metrics["discriminator_loss"])
        history["train_real_score"].append(train_metrics["real_score"])
        history["train_fake_score"].append(train_metrics["fake_score"])
        history["val_real_score"].append(val_metrics["real_score"])
        history["val_fake_score"].append(val_metrics["fake_score"])
        history["train_critic_gap"].append(train_metrics["critic_gap"])
        history["val_critic_gap"].append(val_metrics["critic_gap"])
        history["train_real_class_acc"].append(train_metrics["real_class_accuracy"])
        history["train_fake_class_acc"].append(train_metrics["fake_class_accuracy"])
        history["val_real_class_acc"].append(val_metrics["real_class_accuracy"])
        history["val_fake_class_acc"].append(val_metrics["fake_class_accuracy"])
        history["train_edge_match_loss"].append(train_metrics["feature_match_loss"])
        history["val_edge_match_loss"].append(val_metrics["feature_match_loss"])
        history["train_contrast_gap"].append(0.0)
        history["val_contrast_gap"].append(0.0)
        history["train_diversity_penalty"].append(0.0)
        history["val_diversity_penalty"].append(0.0)
        history["train_detail_score"].append(train_metrics["generator_loss"])
        history["val_detail_score"].append(val_metrics["generator_loss"])
        checkpoint_score = _checkpoint_selection_score(val_metrics, config)
        history["val_checkpoint_score"].append(checkpoint_score)
        should_save = epoch >= config.gan_min_checkpoint_epoch and checkpoint_score < best_val_selection_score
        if should_save:
            best_val_generator_loss = val_metrics["generator_loss"]
            best_val_selection_score = checkpoint_score
            save_gan_checkpoint(save_path, model, class_names, config, best_val_generator_loss, best_val_selection_score, epoch, val_metrics, ema_generator.state_dict())
        if config.gan_preview_every > 0 and (epoch == 1 or epoch % config.gan_preview_every == 0):
            _save_gan_preview(model, preview_labels, fixed_preview_noise, preview_dir / f"epoch_{epoch:04d}.png", ema_generator)
        early_status = "-"
        if early_stopping is not None:
            early_stopping.step(val_metrics["generator_loss"])
            early_status = f"{early_stopping.counter}/{early_stopping.patience}"
        print(f"{epoch:5d} | {train_metrics['generator_loss']:9.4f} | {train_metrics['discriminator_loss']:9.4f} | {val_metrics['generator_loss']:9.4f} | {val_metrics['discriminator_loss']:9.4f} | {val_metrics['real_score']:6.3f} | {val_metrics['fake_score']:6.3f} | {val_metrics['critic_gap']:6.3f} | {val_metrics['feature_match_loss']:7.4f} | {early_status:>12}")
        if early_stopping is not None and early_stopping.should_stop:
            break
    if best_val_selection_score == float("inf"):
        best_val_generator_loss = val_metrics["generator_loss"]
        save_gan_checkpoint(save_path, model, class_names, config, best_val_generator_loss, best_val_generator_loss, epoch, val_metrics, ema_generator.state_dict())
    return history, best_val_generator_loss
