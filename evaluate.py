import json

from config import CONFIG
from src.data import (
    collect_samples_from_directory,
    create_analysis_loader,
    prepare_data,
)
from src.evaluation import (
    calculate_accuracy,
    collect_predictions,
    load_checkpoint,
    make_classification_report,
    make_confusion_matrix,
)
from src.gan_engine import load_gan_checkpoint
from src.models import ClasswiseGANEnsemble, build_gan, build_model
from src.synthetic import (
    build_latent_space_metrics,
    collect_gan_analysis_batches,
    collect_synthetic_feature_distances,
    encode_loader,
    project_latent_space,
    save_feature_metrics_json,
    summarize_feature_metrics,
)
from src.utils import get_device, print_section, set_seed
from src.visualization import (
    plot_confusion_matrix,
    plot_gan_feature_distribution,
    plot_gan_gallery,
    plot_latent_space_distribution,
)


def save_skipped_analysis_note(output_path, reason: str) -> None:
    output_path.write_text(
        json.dumps({"status": "skipped", "reason": reason}, indent=4),
        encoding="utf-8",
    )


def load_classwise_gan_ensemble(config, class_names, device):
    models_by_class = {}
    best_losses = {}

    for class_index, class_name in enumerate(class_names):
        checkpoint_path = config.gan_models_dir / class_name / config.gan_model_filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model GAN untuk kelas '{class_name}' belum ditemukan: {checkpoint_path}."
            )

        class_model = build_gan(config, num_classes=1).to(device)
        checkpoint = load_gan_checkpoint(class_model, checkpoint_path, device)
        class_model.eval()

        models_by_class[class_index] = class_model
        best_losses[class_name] = checkpoint["best_val_generator_loss"]

    ensemble = ClasswiseGANEnsemble(
        models_by_class=models_by_class,
        num_classes=len(class_names),
    ).to(device)
    ensemble.eval()
    return ensemble, best_losses


def main() -> None:
    CONFIG.validate()
    set_seed(CONFIG.seed)

    output_dir = CONFIG.resolved_output_dir
    classifier_checkpoint_path = output_dir / CONFIG.model_filename
    synthetic_root = CONFIG.synthetic_output_dir

    if CONFIG.gan_per_class:
        if not CONFIG.gan_models_dir.exists():
            raise FileNotFoundError(
                f"Folder checkpoint GAN per kelas belum ditemukan: {CONFIG.gan_models_dir}. "
                "Jalankan train.py terlebih dahulu."
            )
    else:
        gan_checkpoint_path = output_dir / CONFIG.gan_model_filename
        if not gan_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model GAN belum ditemukan: {gan_checkpoint_path}. Jalankan train.py terlebih dahulu."
            )

    if not classifier_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model classifier belum ditemukan: {classifier_checkpoint_path}. Jalankan train.py terlebih dahulu."
        )

    if not synthetic_root.exists():
        raise FileNotFoundError(
            f"Folder gambar sintetis belum ditemukan: {synthetic_root}. Jalankan train.py terlebih dahulu."
        )

    device = get_device(CONFIG.device)

    print_section("1. Menyiapkan Dataset")
    data = prepare_data(CONFIG)
    print(f"Folder dataset : {data.dataset_dir}")
    print(f"Nama kelas     : {data.class_names}")
    print(f"Jumlah test    : {data.test_size}")

    print_section("2. Memuat GAN dan Classifier")
    classifier = build_model(
        model_name=CONFIG.model_name,
        num_classes=len(data.class_names),
        dropout=CONFIG.dropout,
        pretrained=False,
        freeze_backbone=False,
        input_channels=CONFIG.image_channels,
    ).to(device)

    if CONFIG.gan_per_class:
        gan, best_loss_by_class = load_classwise_gan_ensemble(
            config=CONFIG,
            class_names=data.class_names,
            device=device,
        )
        print(f"GAN per kelas dimuat dari : {CONFIG.gan_models_dir}")
        for class_name, best_loss in best_loss_by_class.items():
            print(f"Best val loss {class_name:<10}: {best_loss:.6f}")
    else:
        gan = build_gan(CONFIG, num_classes=len(data.class_names)).to(device)
        gan_checkpoint = load_gan_checkpoint(gan, gan_checkpoint_path, device)
        print(f"GAN dimuat dari         : {gan_checkpoint_path}")
        print(f"Best val loss generator : {gan_checkpoint['best_val_generator_loss']:.6f}")

    clf_checkpoint = load_checkpoint(classifier, classifier_checkpoint_path, device)
    print(f"Classifier dimuat dari  : {classifier_checkpoint_path}")
    print(f"Best val acc classifier : {clf_checkpoint['best_val_accuracy']:.2f}%")

    print_section("3. Evaluasi Classifier")
    labels, predictions = collect_predictions(classifier, data.clf_test_loader, device)
    test_accuracy = calculate_accuracy(labels, predictions)
    report = make_classification_report(labels, predictions, data.class_names)
    cm = make_confusion_matrix(labels, predictions)

    report_path = output_dir / "test_report.txt"
    cm_path = output_dir / "confusion_matrix_test.png"
    report_path.write_text(
        f"Test Accuracy: {test_accuracy:.2f}%\n\n{report}",
        encoding="utf-8",
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=data.class_names,
        title="Confusion Matrix - Test Set",
        output_path=cm_path,
    )

    print(f"Test accuracy: {test_accuracy:.2f}%")

    print_section("4. Evaluasi Embedding Space")
    synthetic_samples = collect_samples_from_directory(
        root_dir=synthetic_root,
        class_to_idx=data.class_to_idx,
    )

    latent_plot_path = output_dir / "embedding_distribution.png"
    latent_metrics_path = output_dir / "embedding_metrics.json"
    gan_analysis = collect_gan_analysis_batches(
        model=gan,
        data_loader=create_analysis_loader(
            samples=data.test_samples,
            transform=data.gan_eval_transform,
            batch_size=CONFIG.batch_size,
            num_workers=CONFIG.num_workers,
        ),
        device=device,
        preview_count=CONFIG.gan_preview_count,
        noise_scale=CONFIG.gan_noise_scale,
        config=CONFIG,
    )
    feature_metrics_path = output_dir / "feature_error_metrics.json"
    feature_plot_path = output_dir / "feature_error_distribution.png"
    gan_gallery_path = output_dir / "gan_sample_gallery.png"
    plot_gan_gallery(
        real_images=gan_analysis["preview_real_images"],
        generated_images=gan_analysis["preview_fake_images"],
        labels=gan_analysis["preview_labels"],
        class_names=data.class_names,
        output_path=gan_gallery_path,
    )

    if CONFIG.gan_per_class:
        skip_reason = (
            "Embedding PCA dan feature-distance lintas kelas dinonaktifkan "
            "untuk GAN per kelas karena feature space tiap discriminator "
            "tidak comparable."
        )
        save_skipped_analysis_note(latent_metrics_path, skip_reason)
        save_skipped_analysis_note(feature_metrics_path, skip_reason)
        latent_plot_path.unlink(missing_ok=True)
        feature_plot_path.unlink(missing_ok=True)
        print_section("5. Analisis GAN")
        print("Embedding/feature metrics dilewati karena mode GAN per kelas.")
    else:
        test_analysis_loader = create_analysis_loader(
            samples=data.test_samples,
            transform=data.gan_eval_transform,
            batch_size=CONFIG.batch_size,
            num_workers=CONFIG.num_workers,
        )
        synthetic_analysis_loader = create_analysis_loader(
            samples=synthetic_samples,
            transform=data.gan_eval_transform,
            batch_size=CONFIG.batch_size,
            num_workers=CONFIG.num_workers,
        )

        real_test_embeddings, real_test_labels = encode_loader(
            gan,
            test_analysis_loader,
            device,
        )
        synthetic_embeddings, synthetic_labels = encode_loader(
            gan,
            synthetic_analysis_loader,
            device,
        )

        (
            real_projection,
            real_projection_labels,
            synthetic_projection,
            synthetic_projection_labels,
            explained_variance,
        ) = project_latent_space(
            real_latents=real_test_embeddings,
            real_labels=real_test_labels,
            synthetic_latents=synthetic_embeddings,
            synthetic_labels=synthetic_labels,
            max_points=CONFIG.embedding_vis_max_points,
            seed=CONFIG.seed,
        )

        latent_metrics = build_latent_space_metrics(
            real_latents=real_test_embeddings,
            real_labels=real_test_labels,
            synthetic_latents=synthetic_embeddings,
            synthetic_labels=synthetic_labels,
            class_names=data.class_names,
        )
        latent_metrics["pca_explained_variance"] = explained_variance

        plot_latent_space_distribution(
            real_projection=real_projection,
            real_labels=real_projection_labels,
            synthetic_projection=synthetic_projection,
            synthetic_labels=synthetic_projection_labels,
            class_names=data.class_names,
            explained_variance=explained_variance,
            output_path=latent_plot_path,
        )
        latent_metrics_path.write_text(json.dumps(latent_metrics, indent=4), encoding="utf-8")

        print_section("5. Feature Distance dan GAN Score")
        synthetic_feature_distance = collect_synthetic_feature_distances(
            model=gan,
            data_loader=synthetic_analysis_loader,
            real_latents_by_class=gan_analysis["real_latents_by_class"],
            device=device,
        )

        feature_metrics = summarize_feature_metrics(
            real_scores=gan_analysis["real_scores"],
            synthetic_scores=gan_analysis["synthetic_scores"],
            synthetic_feature_distance=synthetic_feature_distance,
            class_stats=latent_metrics["per_class"],
            class_names=data.class_names,
        )
        save_feature_metrics_json(feature_metrics, feature_metrics_path)
        plot_gan_feature_distribution(
            real_scores=gan_analysis["real_scores"],
            synthetic_scores=gan_analysis["synthetic_scores"],
            synthetic_feature_distance=synthetic_feature_distance,
            output_path=feature_plot_path,
        )

    print_section("6. Output Evaluasi")
    print(f"Report              : {report_path}")
    print(f"Confusion matrix    : {cm_path}")
    print(f"Embedding metrics   : {latent_metrics_path}")
    print(f"Feature error       : {feature_metrics_path}")
    print(f"GAN gallery         : {gan_gallery_path}")


if __name__ == "__main__":
    main()
