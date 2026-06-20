import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(input_channels, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed_gamma = nn.Embedding(num_classes, num_features)
        self.embed_beta = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.embed_gamma.weight)
        nn.init.zeros_(self.embed_beta.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(x)
        gamma = self.embed_gamma(labels).view(-1, self.num_features, 1, 1)
        beta = self.embed_beta(labels).view(-1, self.num_features, 1, 1)
        return normalized * gamma + beta


class ResBlockUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.learnable_skip = in_channels != out_channels
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if self.learnable_skip else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.bn1(x, labels))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.activation(self.bn2(h, labels))
        h = self.conv2(h)
        skip = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
        return h + skip


class ResBlockDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.downsample = downsample
        self.learnable_skip = in_channels != out_channels or downsample
        self.skip_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)) if self.learnable_skip else None
        self.activation = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        skip = x
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
        if self.downsample:
            skip = F.avg_pool2d(skip, 2)
        return h + skip


class ConditionalGenerator(nn.Module):
    def __init__(self, image_channels: int, num_classes: int, image_size: int, latent_dim: int = 128, base_channels: int = 64, class_embedding_dim: int = 32) -> None:
        super().__init__()
        init_size = 4
        num_upsamples = int(math.log2(image_size / init_size))
        if image_size % init_size != 0 or num_upsamples < 2:
            raise ValueError("IMAGE_SIZE harus kelipatan 4 dan minimal 16.")
        multipliers = [min(8, 2 ** i) for i in reversed(range(num_upsamples))]
        self.latent_dim = latent_dim
        self.init_size = init_size
        self.init_channels = base_channels * multipliers[0]
        self.fc = nn.Linear(latent_dim, self.init_channels * init_size * init_size)
        blocks = []
        in_channels = self.init_channels
        for index in range(num_upsamples):
            out_channels = base_channels * (multipliers[index + 1] if index + 1 < len(multipliers) else 1)
            blocks.append(ResBlockUp(in_channels, out_channels, num_classes))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.final_bn = nn.BatchNorm2d(in_channels)
        self.final_conv = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.fc(noise).view(noise.size(0), self.init_channels, self.init_size, self.init_size)
        for block in self.blocks:
            x = block(x, labels)
        x = self.activation(self.final_bn(x))
        return torch.tanh(self.final_conv(x))


class ConditionalDiscriminator(nn.Module):
    def __init__(self, image_channels: int, num_classes: int, image_size: int, base_channels: int = 64, use_spectral_norm: bool = True, use_edge_discriminator: bool = False, edge_discriminator_weight: float = 0.0) -> None:
        super().__init__()
        num_downsamples = int(math.log2(image_size / 4))
        multipliers = [min(8, 2 ** i) for i in range(num_downsamples)]
        blocks = []
        in_channels = image_channels
        for multiplier in multipliers:
            out_channels = base_channels * multiplier
            blocks.append(ResBlockDown(in_channels, out_channels, downsample=True))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.feature_dim = in_channels
        self.linear = nn.utils.spectral_norm(nn.Linear(in_channels, 1))
        self.label_embedding = nn.utils.spectral_norm(nn.Embedding(num_classes, in_channels))
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.class_head = None

    def _forward_features(self, images: torch.Tensor):
        x = images
        maps = []
        for block in self.blocks:
            x = block(x)
            maps.append(x)
        x = self.activation(x)
        pooled = torch.sum(x, dim=(2, 3))
        return pooled, maps

    def encode(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        pooled, _ = self._forward_features(images)
        return pooled

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        pooled, maps = self._forward_features(images)
        logits = self.linear(pooled).squeeze(1)
        if labels is not None:
            logits = logits + torch.sum(self.label_embedding(labels) * pooled, dim=1)
        return logits, None, pooled

    def forward_with_maps(self, images: torch.Tensor, labels: torch.Tensor):
        pooled, maps = self._forward_features(images)
        logits = self.linear(pooled).squeeze(1) + torch.sum(self.label_embedding(labels) * pooled, dim=1)
        return logits, maps


class ConditionalGAN(nn.Module):
    def __init__(self, generator: ConditionalGenerator, discriminator: ConditionalDiscriminator) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.feature_dim = discriminator.feature_dim
        self.latent_dim = generator.latent_dim

    def sample_noise(self, batch_size: int, device: torch.device, noise_scale: float = 1.0) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device) * noise_scale

    def generate(self, labels: torch.Tensor, noise: torch.Tensor | None = None, noise_scale: float = 1.0) -> torch.Tensor:
        if noise is None:
            noise = self.sample_noise(labels.size(0), labels.device, noise_scale)
        return self.generator(noise, labels)

    def discriminate(self, images: torch.Tensor, labels: torch.Tensor):
        return self.discriminator(images, labels)

    def encode(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.discriminator.encode(images, labels)

    def forward(self, images: torch.Tensor, labels: torch.Tensor, noise: torch.Tensor | None = None, noise_scale: float = 1.0):
        generated = self.generate(labels, noise=noise, noise_scale=noise_scale)
        logits, class_logits, features = self.discriminate(generated, labels)
        return generated, logits, class_logits, features


class ClasswiseGANEnsemble(nn.Module):
    def __init__(self, models_by_class: Dict[int, ConditionalGAN], num_classes: int) -> None:
        super().__init__()
        if not models_by_class:
            raise ValueError("models_by_class tidak boleh kosong.")
        self.models = nn.ModuleDict({str(class_index): model for class_index, model in models_by_class.items()})
        self.num_classes = num_classes
        first_model = next(iter(models_by_class.values()))
        self.feature_dim = first_model.feature_dim
        self.latent_dim = first_model.latent_dim

    def sample_noise(self, batch_size: int, device: torch.device, noise_scale: float = 1.0) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device) * noise_scale

    def _build_local_labels(self, labels: torch.Tensor, class_index: int) -> torch.Tensor:
        return torch.zeros(int((labels == class_index).sum().item()), dtype=torch.long, device=labels.device)

    def generate(self, labels: torch.Tensor, noise: torch.Tensor | None = None, noise_scale: float = 1.0) -> torch.Tensor:
        if noise is None:
            noise = self.sample_noise(labels.size(0), labels.device, noise_scale)
        generated_batches = None
        for class_index in labels.unique(sorted=True).tolist():
            mask = labels == class_index
            class_generated = self.models[str(class_index)].generate(self._build_local_labels(labels, class_index), noise=noise[mask], noise_scale=noise_scale)
            if generated_batches is None:
                generated_batches = torch.empty((labels.size(0), *class_generated.shape[1:]), device=class_generated.device, dtype=class_generated.dtype)
            generated_batches[mask] = class_generated
        return generated_batches

    def discriminate(self, images: torch.Tensor, labels: torch.Tensor):
        logits = torch.empty(images.size(0), device=images.device, dtype=images.dtype)
        features = torch.empty(images.size(0), self.feature_dim, device=images.device, dtype=images.dtype)
        class_logits = torch.full((images.size(0), self.num_classes), -1e4, device=images.device, dtype=images.dtype)
        for class_index in labels.unique(sorted=True).tolist():
            mask = labels == class_index
            local_labels = self._build_local_labels(labels, class_index)
            out_logits, _, out_features = self.models[str(class_index)].discriminate(images[mask], local_labels)
            logits[mask] = out_logits
            features[mask] = out_features
            class_logits[mask, class_index] = 1.0
        return logits, class_logits, features

    def encode(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, _, features = self.discriminate(images, labels)
        return features


def freeze_parameters(parameters: Iterable) -> None:
    for parameter in parameters:
        parameter.requires_grad = False


def build_pretrained_model(model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    from torchvision import models
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            freeze_parameters(model.parameters())
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            freeze_parameters(model.parameters())
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            freeze_parameters(model.parameters())
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Pretrained model '{model_name}' belum tersedia.")


def build_model(model_name: str, num_classes: int, dropout: float = 0.5, pretrained: bool = False, freeze_backbone: bool = False, input_channels: int = 3):
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout=dropout, input_channels=input_channels)
    return build_pretrained_model(model_name, num_classes, pretrained, freeze_backbone)


def build_gan(config, num_classes: int) -> ConditionalGAN:
    generator = ConditionalGenerator(
        image_channels=config.image_channels,
        num_classes=num_classes,
        image_size=config.image_size,
        latent_dim=config.gan_latent_dim,
        base_channels=config.gan_base_channels,
        class_embedding_dim=config.gan_class_embedding_dim,
    )
    discriminator = ConditionalDiscriminator(
        image_channels=config.image_channels,
        num_classes=num_classes,
        image_size=config.image_size,
        base_channels=config.gan_base_channels,
        use_spectral_norm=True,
        use_edge_discriminator=False,
        edge_discriminator_weight=0.0,
    )
    return ConditionalGAN(generator=generator, discriminator=discriminator)
