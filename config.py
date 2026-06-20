from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


def get_str(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def get_optional_str(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None or value.strip() == "" or value.lower() in {"none", "null"}:
        return default
    return value.strip()


def get_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def get_optional_int(key: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(key)
    if value is None:
        return default
    if value.strip() == "" or value.lower() in {"none", "null"}:
        return None
    return int(value)


def get_csv(key: str, default: str = "") -> tuple[str, ...]:
    value = os.getenv(key, default)
    return tuple(
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    )


def get_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default

    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False

    raise ValueError(
        f"Nilai boolean untuk {key} tidak valid: {value}. "
        "Gunakan true/false, 1/0, atau yes/no."
    )


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


@dataclass
class Config:
    # 1. DATASET
    sumber_dataset: str = get_str("SUMBER_DATASET", "folder_lokal").lower()
    dataset_root: str = get_str("DATASET_ROOT", "./dataset")
    dataset_name: str = get_str("DATASET_NAME", "banana_ripeness")
    folder_lokal: Optional[str] = get_optional_str("FOLDER_LOKAL", None)
    kaggle_dataset: str = get_str(
        "KAGGLE_DATASET",
        "barkataliarbab/banana-ripeness-classification-classification",
    )
    kaggle_cache_dir: str = get_str("KAGGLE_CACHE_DIR", "./dataset/_kaggle_cache")
    allow_kaggle_download: bool = get_bool("ALLOW_KAGGLE_DOWNLOAD", False)
    max_per_class: Optional[int] = get_optional_int("MAX_PER_CLASS", 400)
    split_by_source: bool = get_bool("SPLIT_BY_SOURCE", True)
    train_ratio: float = get_float("TRAIN_RATIO", 0.70)
    val_ratio: float = get_float("VAL_RATIO", 0.15)
    test_ratio: float = get_float("TEST_RATIO", 0.15)

    # 2. PREPROCESSING GAMBAR
    image_size: int = get_int("IMAGE_SIZE", 128)
    augment_train: bool = get_bool("AUGMENT_TRAIN", True)
    image_channels: int = get_int("IMAGE_CHANNELS", 3)
    crop_foreground: bool = get_bool("CROP_FOREGROUND", True)
    crop_threshold: float = get_float("CROP_THRESHOLD", 28.0)
    crop_margin_ratio: float = get_float("CROP_MARGIN_RATIO", 0.06)

    # 3. TRAINING CLASSIFIER
    batch_size: int = get_int("BATCH_SIZE", 32)
    epochs: int = get_int("EPOCHS", 30)
    learning_rate: float = get_float("LEARNING_RATE", 0.001)
    weight_decay: float = get_float("WEIGHT_DECAY", 0.0)
    num_workers: int = get_int("NUM_WORKERS", 2)
    device: str = get_str("DEVICE", "auto")

    early_stopping: bool = get_bool("EARLY_STOPPING", True)
    early_stopping_patience: int = get_int("EARLY_STOPPING_PATIENCE", 5)
    early_stopping_min_delta: float = get_float("EARLY_STOPPING_MIN_DELTA", 0.001)
    early_stopping_monitor: str = get_str("EARLY_STOPPING_MONITOR", "val_loss")

    # 4. DOWNSTREAM CLASSIFIER
    model_name: str = get_str("MODEL_NAME", "simple_cnn")
    pretrained: bool = get_bool("PRETRAINED", False)
    freeze_backbone: bool = get_bool("FREEZE_BACKBONE", False)
    dropout: float = get_float("DROPOUT", 0.5)

    # 5. GAN
    gan_epochs: int = get_int("GAN_EPOCHS", 60)
    gan_learning_rate_g: float = get_float("GAN_LEARNING_RATE_G", 0.0002)
    gan_learning_rate_d: float = get_float("GAN_LEARNING_RATE_D", 0.0001)
    gan_weight_decay_g: float = get_float("GAN_WEIGHT_DECAY_G", 0.0)
    gan_weight_decay_d: float = get_float("GAN_WEIGHT_DECAY_D", 0.0)
    gan_latent_dim: int = get_int("GAN_LATENT_DIM", 128)
    gan_base_channels: int = get_int("GAN_BASE_CHANNELS", 32)
    gan_class_embedding_dim: int = get_int("GAN_CLASS_EMBEDDING_DIM", 32)
    gan_model_filename: str = get_str("GAN_MODEL_FILENAME", "best_gan.pth")
    gan_augment_train: bool = get_bool("GAN_AUGMENT_TRAIN", False)
    gan_exclude_augmented: bool = get_bool("GAN_EXCLUDE_AUGMENTED", True)
    gan_excluded_augmentation_types: tuple[str, ...] = get_csv(
        "GAN_EXCLUDED_AUGMENTATION_TYPES",
        "blur,noise,brightcontrast,clahe,rotate,hflip",
    )
    gan_beta1: float = get_float("GAN_BETA1", 0.5)
    gan_beta2: float = get_float("GAN_BETA2", 0.999)
    gan_loss_mode: str = get_str("GAN_LOSS_MODE", "hinge").lower()
    gan_gp_weight: float = get_float("GAN_GP_WEIGHT", 10.0)
    gan_drift_weight: float = get_float("GAN_DRIFT_WEIGHT", 0.001)
    gan_r1_gamma: float = get_float("GAN_R1_GAMMA", 5.0)
    gan_r1_every: int = get_int("GAN_R1_EVERY", 16)
    gan_force_gpu: bool = get_bool("GAN_FORCE_GPU", False)
    gan_use_spectral_norm: bool = get_bool("GAN_USE_SPECTRAL_NORM", True)
    gan_use_edge_discriminator: bool = get_bool(
        "GAN_USE_EDGE_DISCRIMINATOR",
        True,
    )
    gan_edge_discriminator_weight: float = get_float(
        "GAN_EDGE_DISCRIMINATOR_WEIGHT",
        0.5,
    )
    gan_ema_decay: float = get_float("GAN_EMA_DECAY", 0.995)
    gan_label_smoothing: float = get_float("GAN_LABEL_SMOOTHING", 0.1)
    gan_feature_match_weight: float = get_float("GAN_FEATURE_MATCH_WEIGHT", 0.0)
    gan_edge_match_weight: float = get_float("GAN_EDGE_MATCH_WEIGHT", 0.0)
    gan_aux_class_weight: float = get_float("GAN_AUX_CLASS_WEIGHT", 0.5)
    gan_disc_steps: int = get_int("GAN_DISC_STEPS", 1)
    gan_early_stopping: bool = get_bool("GAN_EARLY_STOPPING", False)
    gan_min_checkpoint_epoch: int = get_int("GAN_MIN_CHECKPOINT_EPOCH", 20)
    gan_preview_every: int = get_int("GAN_PREVIEW_EVERY", 10)
    gan_preview_grid_count: int = get_int("GAN_PREVIEW_GRID_COUNT", 16)
    gan_per_class: bool = get_bool("GAN_PER_CLASS", True)
    gan_models_dirname: str = get_str("GAN_MODELS_DIRNAME", "classwise_gans")

    # 6. GENERASI DATASET SINTETIS
    generated_images_per_class: int = get_int("GENERATED_IMAGES_PER_CLASS", 100)
    gan_noise_scale: float = get_float("GAN_NOISE_SCALE", 1.0)
    synthetic_output_dirname: str = get_str(
        "SYNTHETIC_OUTPUT_DIRNAME",
        "generate_synthetic_image",
    )
    embedding_vis_max_points: int = get_int("EMBEDDING_VIS_MAX_POINTS", 1500)
    gan_preview_count: int = get_int("GAN_PREVIEW_COUNT", 8)

    # 7. OUTPUT
    output_dir: str = get_str("OUTPUT_DIR", "outputs")
    model_filename: str = get_str("MODEL_FILENAME", "best_model.pth")
    seed: int = get_int("SEED", 42)

    @property
    def dataset_path(self) -> Path:
        if self.folder_lokal:
            return resolve_project_path(self.folder_lokal)
        return resolve_project_path(Path(self.dataset_root) / self.dataset_name)

    @property
    def resolved_output_dir(self) -> Path:
        return resolve_project_path(self.output_dir)

    @property
    def resolved_kaggle_cache_dir(self) -> Path:
        return resolve_project_path(self.kaggle_cache_dir)

    @property
    def synthetic_output_dir(self) -> Path:
        return self.resolved_output_dir / self.synthetic_output_dirname

    @property
    def gan_models_dir(self) -> Path:
        return self.resolved_output_dir / self.gan_models_dirname

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> dict:
        return {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.resolved_output_dir),
            "synthetic_output_dir": str(self.synthetic_output_dir),
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "image_size": self.image_size,
            "split_by_source": self.split_by_source,
            "crop_foreground": self.crop_foreground,
            "crop_threshold": self.crop_threshold,
            "crop_margin_ratio": self.crop_margin_ratio,
            "batch_size": self.batch_size,
            "epochs_classifier": self.epochs,
            "epochs_gan": self.gan_epochs,
            "gan_latent_dim": self.gan_latent_dim,
            "gan_base_channels": self.gan_base_channels,
            "gan_augment_train": self.gan_augment_train,
            "gan_exclude_augmented": self.gan_exclude_augmented,
            "gan_excluded_augmentation_types": self.gan_excluded_augmentation_types,
            "gan_label_smoothing": self.gan_label_smoothing,
            "gan_loss_mode": self.gan_loss_mode,
            "gan_gp_weight": self.gan_gp_weight,
            "gan_drift_weight": self.gan_drift_weight,
            "gan_r1_gamma": self.gan_r1_gamma,
            "gan_r1_every": self.gan_r1_every,
            "gan_force_gpu": self.gan_force_gpu,
            "gan_use_spectral_norm": self.gan_use_spectral_norm,
            "gan_use_edge_discriminator": self.gan_use_edge_discriminator,
            "gan_edge_discriminator_weight": self.gan_edge_discriminator_weight,
            "gan_ema_decay": self.gan_ema_decay,
            "gan_feature_match_weight": self.gan_feature_match_weight,
            "gan_edge_match_weight": self.gan_edge_match_weight,
            "gan_aux_class_weight": self.gan_aux_class_weight,
            "gan_min_checkpoint_epoch": self.gan_min_checkpoint_epoch,
            "gan_preview_every": self.gan_preview_every,
            "gan_preview_grid_count": self.gan_preview_grid_count,
            "gan_early_stopping": self.gan_early_stopping,
            "gan_per_class": self.gan_per_class,
            "gan_models_dir": str(self.gan_models_dir),
            "generated_images_per_class": self.generated_images_per_class,
            "gan_noise_scale": self.gan_noise_scale,
            "device": self.device,
            "seed": self.seed,
        }

    def validate(self) -> None:
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "TRAIN_RATIO + VAL_RATIO + TEST_RATIO harus sama dengan 1.0"
            )

        if self.sumber_dataset not in {"folder_lokal", "kaggle"}:
            raise ValueError("SUMBER_DATASET harus 'folder_lokal' atau 'kaggle'.")

        if self.sumber_dataset == "folder_lokal" and not self.dataset_path.exists():
            raise FileNotFoundError(
                "Dataset lokal tidak ditemukan.\n"
                f"Path yang dibaca: {self.dataset_path}\n"
                "Solusi: ubah DATASET_ROOT/DATASET_NAME atau FOLDER_LOKAL di .env."
            )

        if self.sumber_dataset == "kaggle" and not self.allow_kaggle_download:
            raise RuntimeError(
                "SUMBER_DATASET=kaggle, tetapi ALLOW_KAGGLE_DOWNLOAD=false.\n"
                "Ubah SUMBER_DATASET=folder_lokal atau set ALLOW_KAGGLE_DOWNLOAD=true."
            )

        if self.model_name not in {
            "simple_cnn",
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
        }:
            raise ValueError(f"MODEL_NAME '{self.model_name}' belum tersedia.")

        if (
            self.image_size < 64
            or self.image_size % 16 != 0
            or self.image_size & (self.image_size - 1)
        ):
            raise ValueError(
                "IMAGE_SIZE harus berupa pangkat dua, minimal 64, dan habis "
                "dibagi 16 agar arsitektur GAN tetap sinkron."
            )

        if self.crop_threshold <= 0:
            raise ValueError("CROP_THRESHOLD harus lebih besar dari 0.")

        if not (0.0 <= self.crop_margin_ratio < 0.5):
            raise ValueError("CROP_MARGIN_RATIO harus berada pada rentang [0.0, 0.5).")

        if self.generated_images_per_class <= 0:
            raise ValueError("GENERATED_IMAGES_PER_CLASS harus lebih besar dari 0.")

        if self.gan_latent_dim <= 0:
            raise ValueError("GAN_LATENT_DIM harus lebih besar dari 0.")

        if self.gan_disc_steps <= 0:
            raise ValueError("GAN_DISC_STEPS harus lebih besar dari 0.")

        if self.gan_min_checkpoint_epoch <= 0:
            raise ValueError("GAN_MIN_CHECKPOINT_EPOCH harus lebih besar dari 0.")

        if self.gan_preview_every < 0:
            raise ValueError("GAN_PREVIEW_EVERY tidak boleh negatif.")

        if self.gan_preview_grid_count <= 0:
            raise ValueError("GAN_PREVIEW_GRID_COUNT harus lebih besar dari 0.")

        if self.gan_loss_mode not in {"hinge", "bce", "wgan_gp"}:
            raise ValueError("GAN_LOSS_MODE harus 'hinge', 'bce', atau 'wgan_gp'.")

        if self.gan_r1_gamma < 0:
            raise ValueError("GAN_R1_GAMMA tidak boleh negatif.")

        if self.gan_r1_every < 0:
            raise ValueError("GAN_R1_EVERY tidak boleh negatif.")

        if self.gan_force_gpu and self.device.lower() == "cpu":
            raise ValueError("GAN_FORCE_GPU=true tidak kompatibel dengan DEVICE=cpu.")

        if self.gan_gp_weight < 0:
            raise ValueError("GAN_GP_WEIGHT tidak boleh negatif.")

        if self.gan_drift_weight < 0:
            raise ValueError("GAN_DRIFT_WEIGHT tidak boleh negatif.")

        if self.gan_edge_discriminator_weight < 0:
            raise ValueError("GAN_EDGE_DISCRIMINATOR_WEIGHT tidak boleh negatif.")

        if not (0.0 <= self.gan_ema_decay < 1.0):
            raise ValueError("GAN_EMA_DECAY harus berada pada rentang [0.0, 1.0).")

        if not (0.0 <= self.gan_label_smoothing < 1.0):
            raise ValueError("GAN_LABEL_SMOOTHING harus berada pada rentang [0.0, 1.0).")

        if self.gan_noise_scale <= 0:
            raise ValueError("GAN_NOISE_SCALE harus lebih besar dari 0.")

        if self.gan_feature_match_weight < 0:
            raise ValueError("GAN_FEATURE_MATCH_WEIGHT tidak boleh negatif.")

        if self.gan_edge_match_weight < 0:
            raise ValueError("GAN_EDGE_MATCH_WEIGHT tidak boleh negatif.")

        if self.gan_aux_class_weight < 0:
            raise ValueError("GAN_AUX_CLASS_WEIGHT tidak boleh negatif.")


CONFIG = Config()
