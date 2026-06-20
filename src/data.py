import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GAN_MEAN = [0.5, 0.5, 0.5]
GAN_STD = [0.5, 0.5, 0.5]
AUGMENTED_PREFIX_PATTERN = re.compile(r"^aug_[a-z0-9_]+?_\d+_", re.IGNORECASE)
COPY_SUFFIX_PATTERN = re.compile(r"[-_\s]*copy(?:\s*\(\d+\))?$", re.IGNORECASE)
PAREN_NUMBER_SUFFIX_PATTERN = re.compile(r"\(\d+\)$")


@dataclass
class DataBundle:
    gan_train_loader: DataLoader
    gan_val_loader: DataLoader
    gan_test_loader: DataLoader
    clf_train_loader: DataLoader
    clf_val_loader: DataLoader
    clf_test_loader: DataLoader
    gan_train_transform: transforms.Compose
    gan_eval_transform: transforms.Compose
    clf_train_transform: transforms.Compose
    class_names: List[str]
    class_to_idx: Dict[str, int]
    dataset_dir: Path
    total_images: int
    train_size: int
    val_size: int
    test_size: int
    train_samples: List[Tuple[Path, int]]
    val_samples: List[Tuple[Path, int]]
    test_samples: List[Tuple[Path, int]]
    gan_train_samples: List[Tuple[Path, int]]
    gan_val_samples: List[Tuple[Path, int]]


def download_kaggle_dataset(config) -> Path:
    if config.sumber_dataset != "kaggle":
        raise RuntimeError(
            "download_kaggle_dataset() terpanggil padahal SUMBER_DATASET bukan kaggle."
        )

    if not config.allow_kaggle_download:
        raise RuntimeError(
            "Download Kaggle diblokir. Set ALLOW_KAGGLE_DOWNLOAD=true jika diperlukan."
        )

    try:
        import kagglehub
    except ImportError as error:
        raise ImportError(
            "Package kagglehub belum terinstall. Jalankan: pip install kagglehub"
        ) from error

    config.resolved_kaggle_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(config.resolved_kaggle_cache_dir)

    print(f"Mengunduh dataset Kaggle: {config.kaggle_dataset}")
    print(f"Cache KaggleHub        : {config.resolved_kaggle_cache_dir}")

    dataset_path = kagglehub.dataset_download(config.kaggle_dataset)
    return Path(dataset_path)


def folder_contains_images(folder: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in folder.iterdir()
    )


def find_class_folder(base_dir: str | Path) -> Path:
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {base_dir}")

    direct_class_dirs = [
        child
        for child in base_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".") and folder_contains_images(child)
    ]
    if len(direct_class_dirs) >= 2:
        return base_dir

    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        class_dirs = []

        for dirname in sorted(dirs):
            class_path = root_path / dirname
            if folder_contains_images(class_path):
                class_dirs.append(dirname)

        root_has_images = any(
            Path(filename).suffix.lower() in IMAGE_EXTENSIONS for filename in files
        )

        if len(class_dirs) >= 2 and not root_has_images:
            return root_path

    raise ValueError(
        "Tidak ditemukan struktur dataset klasifikasi gambar.\n"
        f"Folder yang diperiksa: {base_dir}\n"
        "Struktur yang diharapkan: satu sub-folder untuk setiap kelas.\n"
        "Contoh: ./dataset/banana_ripeness/ripe/*.jpg"
    )


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image

        target_size = max(width, height)
        corner_pixels = [
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        ]
        background = tuple(
            int(sum(channel_values) / len(channel_values))
            for channel_values in zip(*corner_pixels)
        )

        padded = Image.new("RGB", (target_size, target_size), background)
        offset_x = (target_size - width) // 2
        offset_y = (target_size - height) // 2
        padded.paste(image, (offset_x, offset_y))
        return padded


class AutoForegroundCrop:
    def __init__(
        self,
        threshold: float = 28.0,
        margin_ratio: float = 0.06,
        min_foreground_ratio: float = 0.02,
    ) -> None:
        self.threshold = threshold
        self.margin_ratio = margin_ratio
        self.min_foreground_ratio = min_foreground_ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        image_array = np.asarray(image, dtype=np.float32)
        height, width = image_array.shape[:2]

        corners = np.asarray(
            [
                image_array[0, 0],
                image_array[0, width - 1],
                image_array[height - 1, 0],
                image_array[height - 1, width - 1],
            ],
            dtype=np.float32,
        )
        background = corners.mean(axis=0)
        color_distance = np.linalg.norm(image_array - background, axis=2)
        foreground_mask = color_distance > self.threshold

        corner_luminance = (
            0.299 * corners[:, 0] + 0.587 * corners[:, 1] + 0.114 * corners[:, 2]
        ).mean()
        if corner_luminance < 35.0:
            luminance = (
                0.299 * image_array[:, :, 0]
                + 0.587 * image_array[:, :, 1]
                + 0.114 * image_array[:, :, 2]
            )
            brightness_threshold = max(self.threshold, corner_luminance + self.threshold)
            foreground_mask = foreground_mask | (luminance > brightness_threshold)

        foreground_ratio = float(foreground_mask.mean())
        if foreground_ratio < self.min_foreground_ratio:
            return image

        ys, xs = np.where(foreground_mask)
        if len(xs) == 0 or len(ys) == 0:
            return image

        margin = int(max(height, width) * self.margin_ratio)
        left = max(int(xs.min()) - margin, 0)
        top = max(int(ys.min()) - margin, 0)
        right = min(int(xs.max()) + margin + 1, width)
        bottom = min(int(ys.max()) + margin + 1, height)

        if right - left < 8 or bottom - top < 8:
            return image

        return image.crop((left, top, right, bottom))


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        max_per_class: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.samples: List[Tuple[Path, int]] = []

        self.class_names = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir() and not folder.name.startswith(".")
        )

        if not self.class_names:
            raise ValueError(f"Tidak ditemukan sub-folder kelas pada: {self.root_dir}")

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        rng = random.Random(seed)

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            image_paths = [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

            rng.shuffle(image_paths)

            if max_per_class is not None:
                image_paths = image_paths[:max_per_class]

            label = self.class_to_idx[class_name]
            self.samples.extend((path, label) for path in image_paths)

        if not self.samples:
            raise ValueError(f"Tidak ditemukan file gambar pada: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = load_image(image_path)
        return image, label


class ImagePathDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform,
        return_path: bool = False,
    ) -> None:
        self.samples = [(Path(path), label) for path, label in samples]
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, str(image_path)

        return image, label


def build_base_image_ops(
    image_size: int,
    crop_foreground: bool = True,
    crop_threshold: float = 28.0,
    crop_margin_ratio: float = 0.06,
):
    base_ops = []
    if crop_foreground:
        base_ops.append(
            AutoForegroundCrop(
                threshold=crop_threshold,
                margin_ratio=crop_margin_ratio,
            )
        )
    base_ops.extend(
        [
            PadToSquare(),
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]
    )
    return base_ops


def build_transforms(
    image_size: int,
    augment_train: bool = True,
    crop_foreground: bool = True,
    crop_threshold: float = 28.0,
    crop_margin_ratio: float = 0.06,
):
    square_resize_ops = build_base_image_ops(
        image_size=image_size,
        crop_foreground=crop_foreground,
        crop_threshold=crop_threshold,
        crop_margin_ratio=crop_margin_ratio,
    )

    classifier_augmentation_steps = []
    if augment_train:
        classifier_augmentation_steps = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
        ]

    gan_train_transform = transforms.Compose(
        [
            *square_resize_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=GAN_MEAN, std=GAN_STD),
        ]
    )

    gan_eval_transform = transforms.Compose(
        [
            *square_resize_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=GAN_MEAN, std=GAN_STD),
        ]
    )

    clf_train_transform = transforms.Compose(
        [
            *square_resize_ops,
            *classifier_augmentation_steps,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    clf_eval_transform = transforms.Compose(
        [
            *square_resize_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return gan_train_transform, gan_eval_transform, clf_train_transform, clf_eval_transform


def get_dataset_dir(config) -> Path:
    if config.sumber_dataset == "folder_lokal":
        base_dir = config.dataset_path
        print("Mode dataset        : folder_lokal")
        print(f"Dataset aktif       : {config.dataset_name}")
        print(f"Path dataset lokal  : {base_dir}")
    elif config.sumber_dataset == "kaggle":
        base_dir = download_kaggle_dataset(config)
        print("Mode dataset        : kaggle")
        print(f"Path dataset Kaggle : {base_dir}")
    else:
        raise ValueError("SUMBER_DATASET harus 'folder_lokal' atau 'kaggle'.")

    class_folder = find_class_folder(base_dir)
    print(f"Folder kelas terbaca: {class_folder}")
    return class_folder


def subset_to_samples(subset) -> List[Tuple[Path, int]]:
    return [subset.dataset.samples[index] for index in subset.indices]


def source_group_key(image_path: Path) -> str:
    stem = image_path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]

    stem = AUGMENTED_PREFIX_PATTERN.sub("", stem)
    stem = COPY_SUFFIX_PATTERN.sub("", stem)
    stem = PAREN_NUMBER_SUFFIX_PATTERN.sub("", stem)
    return stem.lower()


def augmentation_type(image_path: Path) -> Optional[str]:
    match = re.match(r"^aug_([a-z0-9]+)_", image_path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else None


def filter_gan_samples(
    samples: List[Tuple[Path, int]],
    exclude_augmented: bool,
    excluded_augmentation_types: Tuple[str, ...],
) -> List[Tuple[Path, int]]:
    excluded_types = {item.lower() for item in excluded_augmentation_types}
    filtered_samples = []

    for path, label in samples:
        sample_augmentation = augmentation_type(Path(path))
        if sample_augmentation is None:
            filtered_samples.append((path, label))
            continue

        if exclude_augmented or sample_augmentation in excluded_types:
            continue

        filtered_samples.append((path, label))

    return filtered_samples


def split_samples_by_source(
    samples: List[Tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    grouped_by_label: Dict[int, Dict[str, List[Tuple[Path, int]]]] = {}

    for path, label in samples:
        grouped_by_label.setdefault(label, {}).setdefault(
            source_group_key(Path(path)),
            [],
        ).append((path, label))

    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    test_samples: List[Tuple[Path, int]] = []

    for label in sorted(grouped_by_label):
        groups = list(grouped_by_label[label].values())
        rng.shuffle(groups)

        total_images = sum(len(group) for group in groups)
        target_train = int(total_images * train_ratio)
        target_val = int(total_images * val_ratio)

        class_train: List[Tuple[Path, int]] = []
        class_val: List[Tuple[Path, int]] = []
        class_test: List[Tuple[Path, int]] = []

        for group in groups:
            if len(class_train) < target_train:
                class_train.extend(group)
            elif len(class_val) < target_val:
                class_val.extend(group)
            else:
                class_test.extend(group)

        if not class_train or not class_val or not class_test:
            raise ValueError(
                "Split berbasis source gagal karena jumlah source per kelas terlalu kecil. "
                "Coba set SPLIT_BY_SOURCE=false atau ubah rasio split."
            )

        train_samples.extend(class_train)
        val_samples.extend(class_val)
        test_samples.extend(class_test)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


def create_loader(
    samples: List[Tuple[Path, int]],
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    return_path: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    dataset = ImagePathDataset(
        samples=samples,
        transform=transform,
        return_path=return_path,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def collect_samples_from_directory(
    root_dir: str | Path,
    class_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    root_dir = Path(root_dir)
    samples: List[Tuple[Path, int]] = []

    if not root_dir.exists():
        return samples

    for class_name, label in class_to_idx.items():
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue

        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    return samples


def select_samples_for_class(
    samples: List[Tuple[Path, int]],
    target_label: int,
    remap_label: int = 0,
) -> List[Tuple[Path, int]]:
    return [
        (path, remap_label)
        for path, label in samples
        if label == target_label
    ]


def build_classifier_train_loader_with_synthetic(
    config,
    data: DataBundle,
    synthetic_root: str | Path,
) -> Tuple[DataLoader, int]:
    synthetic_samples = collect_samples_from_directory(
        root_dir=synthetic_root,
        class_to_idx=data.class_to_idx,
    )
    combined_samples = list(data.train_samples) + synthetic_samples

    loader = create_loader(
        samples=combined_samples,
        transform=data.clf_train_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    return loader, len(synthetic_samples)


def create_analysis_loader(
    samples: List[Tuple[Path, int]],
    transform,
    batch_size: int,
    num_workers: int,
    return_path: bool = False,
) -> DataLoader:
    return create_loader(
        samples=samples,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        return_path=return_path,
    )


def prepare_data(config) -> DataBundle:
    dataset_dir = get_dataset_dir(config)

    full_dataset = ImageFolderDataset(
        root_dir=dataset_dir,
        max_per_class=config.max_per_class,
        seed=config.seed,
    )

    total_images = len(full_dataset)
    if config.split_by_source:
        train_samples, val_samples, test_samples = split_samples_by_source(
            samples=full_dataset.samples,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        print("Strategi split      : source_group")
    else:
        train_size = int(total_images * config.train_ratio)
        val_size = int(total_images * config.val_ratio)
        test_size = total_images - train_size - val_size

        if min(train_size, val_size, test_size) <= 0:
            raise ValueError(
                "Jumlah data terlalu kecil untuk split train/val/test. "
                "Kurangi rasio split atau tambahkan gambar."
            )

        generator = torch.Generator().manual_seed(config.seed)
        train_raw, val_raw, test_raw = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

        train_samples = subset_to_samples(train_raw)
        val_samples = subset_to_samples(val_raw)
        test_samples = subset_to_samples(test_raw)
        print("Strategi split      : random_file")

    train_size = len(train_samples)
    val_size = len(val_samples)
    test_size = len(test_samples)

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "Jumlah data terlalu kecil untuk split train/val/test. "
            "Kurangi rasio split atau tambahkan gambar."
        )

    (
        gan_train_transform,
        gan_eval_transform,
        clf_train_transform,
        clf_eval_transform,
    ) = build_transforms(
        image_size=config.image_size,
        augment_train=config.augment_train,
        crop_foreground=config.crop_foreground,
        crop_threshold=config.crop_threshold,
        crop_margin_ratio=config.crop_margin_ratio,
    )

    if config.gan_augment_train:
        gan_base_ops = build_base_image_ops(
            image_size=config.image_size,
            crop_foreground=config.crop_foreground,
            crop_threshold=config.crop_threshold,
            crop_margin_ratio=config.crop_margin_ratio,
        )
        gan_train_transform = transforms.Compose(
            [
                *gan_base_ops,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=GAN_MEAN, std=GAN_STD),
            ]
        )

    gan_train_samples = filter_gan_samples(
        samples=train_samples,
        exclude_augmented=config.gan_exclude_augmented,
        excluded_augmentation_types=config.gan_excluded_augmentation_types,
    )
    gan_val_samples = filter_gan_samples(
        samples=val_samples,
        exclude_augmented=config.gan_exclude_augmented,
        excluded_augmentation_types=config.gan_excluded_augmentation_types,
    )
    if not gan_train_samples or not gan_val_samples:
        raise ValueError(
            "Filter data GAN menghapus seluruh sampel train/val. "
            "Periksa GAN_EXCLUDE_AUGMENTED dan GAN_EXCLUDED_AUGMENTATION_TYPES."
        )

    removed_train = len(train_samples) - len(gan_train_samples)
    removed_val = len(val_samples) - len(gan_val_samples)
    print(
        "Data GAN bersih     : "
        f"train={len(gan_train_samples)} (-{removed_train}), "
        f"val={len(gan_val_samples)} (-{removed_val})"
    )

    gan_train_loader = create_loader(
        samples=gan_train_samples,
        transform=gan_train_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
    )
    gan_val_loader = create_loader(
        samples=gan_val_samples,
        transform=gan_eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    gan_test_loader = create_loader(
        samples=test_samples,
        transform=gan_eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    clf_train_loader = create_loader(
        samples=train_samples,
        transform=clf_train_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    clf_val_loader = create_loader(
        samples=val_samples,
        transform=clf_eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    clf_test_loader = create_loader(
        samples=test_samples,
        transform=clf_eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    return DataBundle(
        gan_train_loader=gan_train_loader,
        gan_val_loader=gan_val_loader,
        gan_test_loader=gan_test_loader,
        clf_train_loader=clf_train_loader,
        clf_val_loader=clf_val_loader,
        clf_test_loader=clf_test_loader,
        gan_train_transform=gan_train_transform,
        gan_eval_transform=gan_eval_transform,
        clf_train_transform=clf_train_transform,
        class_names=full_dataset.class_names,
        class_to_idx=full_dataset.class_to_idx,
        dataset_dir=dataset_dir,
        total_images=total_images,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        gan_train_samples=gan_train_samples,
        gan_val_samples=gan_val_samples,
    )
