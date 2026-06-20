import json

import torch
import torch.nn as nn

from config import CONFIG
from src.data import (
    build_classifier_train_loader_with_synthetic,
    collect_samples_from_directory,
    create_analysis_loader,
    create_loader,
    prepare_data,
    select_samples_for_class,
)
from src.engine import train_model
from src.evaluation import (
    calculate_accuracy,
    collect_predictions,
    load_checkpoint,
    make_classification_report,
    make_confusion_matrix,
)
from src.gan_engine import load_gan_checkpoint, train_gan
from src.model_graph import export_model_graph
from src.model_report import export_model_report
from src.models import ClasswiseGANEnsemble, build_gan, build_model
from src.synthetic import (
    build_latent_space_metrics,
    collect_gan_analysis_batches,
    collect_synthetic_feature_distances,
    compute_latent_statistics,
    encode_loader,
    generate_synthetic_images,
    project_latent_space,
    save_feature_metrics_json,
    save_latent_statistics_json,
    summarize_feature_metrics,
)
from src.utils import get_device, print_section, save_json, set_seed
from src.visualization import (
    plot_confusion_matrix,
    plot_gan_feature_distribution,
    plot_gan_gallery,
    plot_gan_history,
    plot_history,
    plot_latent_space_distribution,
)


def save_latent_metrics_bundle(metrics, output_path) -> None:
    output_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")


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
                f"Checkpoint GAN untuk kelas '{class_name}' tidak ditemukan: {checkpoint_path}"
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


def train_classwise_gans(config, data, device, gan_assets_dir):
    source_criterion = (
        nn.BCEWithLogitsLoss()
        if config.gan_loss_mode == "bce"
        else None
    )
    class_criterion = nn.CrossEntropyLoss()
    history_summary = {}
    best_loss_summary = {}

    config.gan_models_dir.mkdir(parents=True, exist_ok=True)
    gan_assets_dir.mkdir(parents=True, exist_ok=True)

    for class_index, class_name in enumerate(data.class_names):
        class_train_samples = select_samples_for_class(
            data.gan_train_samples,
            class_index,
        )
        class_val_samples = select_samples_for_class(
            data.gan_val_samples,
            class_index,
        )

        if not class_train_samples or not class_val_samples:
            raise ValueError(
                f"Data train/val untuk kelas '{class_name}' tidak cukup "
                "untuk melatih GAN per kelas."
            )

        print()
        print_section(f"Training GAN - {class_name}")
        print(f"Sampel train kelas {class_name}: {len(class_train_samples)}")
        print(f"Sampel val kelas {class_name}  : {len(class_val_samples)}")

        class_gan = build_gan(config, num_classes=1).to(device)
        class_assets_dir = gan_assets_dir / class_name
        export_model_report(model=class_gan.generator, output_dir=class_assets_dir / "generator")
        export_model_report(
            model=class_gan.discriminator,
            output_dir=class_assets_dir / "discriminator",
        )

        train_loader = create_loader(
            samples=class_train_samples,
            transform=data.gan_train_transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=True,
        )
        val_loader = create_loader(
            samples=class_val_samples,
            transform=data.gan_eval_transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

        generator_optimizer = torch.optim.Adam(
            class_gan.generator.parameters(),
            lr=config.gan_learning_rate_g,
            betas=(config.gan_beta1, config.gan_beta2),
            weight_decay=config.gan_weight_decay_g,
        )
        discriminator_optimizer = torch.optim.Adam(
            class_gan.discriminator.parameters(),
            lr=config.gan_learning_rate_d,
            betas=(config.gan_beta1, config.gan_beta2),
            weight_decay=config.gan_weight_decay_d,
        )

        checkpoint_dir = config.gan_models_dir / class_name
        checkpoint_path = checkpoint_dir / config.gan_model_filename
        history_path = checkpoint_dir / "gan_training_history.json"
        curve_path = checkpoint_dir / "gan_training_curve.png"

        history, best_val_generator_loss = train_gan(
            model=class_gan,
            train_loader=train_loader,
            val_loader=val_loader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            source_criterion=source_criterion,
            class_criterion=class_criterion,
            device=device,
            epochs=config.gan_epochs,
            save_path=checkpoint_path,
            class_names=[class_name],
            config=config,
        )

        save_json(history, history_path)
        plot_gan_history(history, curve_path)
        history_summary[class_name] = history
        best_loss_summary[class_name] = best_val_generator_loss

        print(f"Checkpoint kelas {class_name}: {checkpoint_path}")
        print(f"Best validation loss G      : {best_val_generator_loss:.6f}")

    summary_path = config.resolved_output_dir / "classwise_gan_summary.json"
    save_json(
        {
            "best_val_generator_loss": best_loss_summary,
            "checkpoint_root": str(config.gan_models_dir),
        },
        summary_path,
    )
    return history_summary, best_loss_summary, summary_path


def evaluate_classifier(model, test_loader, class_names, output_dir, device) -> dict:
    labels, predictions = collect_predictions(model, test_loader, device)
    test_accuracy = calculate_accuracy(labels, predictions)
    report = make_classification_report(labels, predictions, class_names)
    cm = make_confusion_matrix(labels, predictions)

    report_path = output_dir / "test_report.txt"
    cm_path = output_dir / "confusion_matrix_test.png"

    report_path.write_text(
        f"Test Accuracy: {test_accuracy:.2f}%\n\n{report}",
        encoding="utf-8",
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title="Confusion Matrix - Test Set",
        output_path=cm_path,
    )

    return {
        "test_accuracy": test_accuracy,
        "report": report,
        "report_path": report_path,
        "cm_path": cm_path,
    }


def main() -> None:
    CONFIG.validate()
    set_seed(CONFIG.seed)

    output_dir = CONFIG.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gan_assets_dir = output_dir / "gan_assets"
    classifier_assets_dir = output_dir / "classifier_assets"
    generator_assets_dir = gan_assets_dir / "generator"
    discriminator_assets_dir = gan_assets_dir / "discriminator"

    device = get_device(CONFIG.device)
    if CONFIG.gan_force_gpu and device.type != "cuda":
        raise RuntimeError("CUDA tidak terdeteksi, padahal GAN_FORCE_GPU=true.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_section("1. Menyiapkan Dataset")
    data = prepare_data(CONFIG)

    print(f"Folder dataset : {data.dataset_dir}")
    print(f"Nama kelas     : {data.class_names}")
    print(f"Total gambar   : {data.total_images}")
    print(
        f"Distribusi     : "
        f"train={data.train_size}, val={data.val_size}, test={data.test_size}"
    )

    print_section("2. Training GAN")
    print(
        "GAN config              : "
        f"latent_dim={CONFIG.gan_latent_dim}, "
        f"base_channels={CONFIG.gan_base_channels}, "
        f"disc_steps={CONFIG.gan_disc_steps}, "
        f"per_class={CONFIG.gan_per_class}"
    )
    print(
        "GAN detail pipeline      : "
        f"exclude_augmented={CONFIG.gan_exclude_augmented}, "
        f"edge_discriminator={CONFIG.gan_use_edge_discriminator}, "
        f"spectral_norm={CONFIG.gan_use_spectral_norm}, "
        f"ema={CONFIG.gan_ema_decay}"
    )
    print(
        "GAN loss stabilizer     : "
        f"loss_mode={CONFIG.gan_loss_mode}, "
        f"label_smoothing={CONFIG.gan_label_smoothing}, "
        f"gp_weight={CONFIG.gan_gp_weight}, "
        f"feature_match={CONFIG.gan_feature_match_weight}, "
        f"aux_class={CONFIG.gan_aux_class_weight}"
    )

    if CONFIG.gan_per_class:
        (
            gan_history,
            best_val_generator_loss,
            gan_summary_path,
        ) = train_classwise_gans(
            config=CONFIG,
            data=data,
            device=device,
            gan_assets_dir=gan_assets_dir,
        )
        gan, best_loss_by_class = load_classwise_gan_ensemble(
            config=CONFIG,
            class_names=data.class_names,
            device=device,
        )
        gan_history_path = gan_summary_path
        gan_curve_path = CONFIG.gan_models_dir
        print("Mode training GAN       : per kelas")
        print(f"Root checkpoint GAN     : {CONFIG.gan_models_dir}")
        print(f"Ringkasan checkpoint    : {gan_summary_path}")
        for class_name, best_loss in best_loss_by_class.items():
            print(f"Best val loss {class_name:<10}: {best_loss:.6f}")
    else:
        gan = build_gan(CONFIG, num_classes=len(data.class_names)).to(device)

        print("Generator report")
        export_model_report(model=gan.generator, output_dir=generator_assets_dir)
        print("\nDiscriminator report")
        export_model_report(model=gan.discriminator, output_dir=discriminator_assets_dir)

        gan_checkpoint_path = output_dir / CONFIG.gan_model_filename
        gan_criterion = nn.BCEWithLogitsLoss() if CONFIG.gan_loss_mode == "bce" else None
        gan_class_criterion = nn.CrossEntropyLoss()
        generator_optimizer = torch.optim.Adam(
            gan.generator.parameters(),
            lr=CONFIG.gan_learning_rate_g,
            betas=(CONFIG.gan_beta1, CONFIG.gan_beta2),
            weight_decay=CONFIG.gan_weight_decay_g,
        )
        discriminator_optimizer = torch.optim.Adam(
            gan.discriminator.parameters(),
            lr=CONFIG.gan_learning_rate_d,
            betas=(CONFIG.gan_beta1, CONFIG.gan_beta2),
            weight_decay=CONFIG.gan_weight_decay_d,
        )

        gan_history, best_val_generator_loss = train_gan(
            model=gan,
            train_loader=data.gan_train_loader,
            val_loader=data.gan_val_loader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            source_criterion=gan_criterion,
            class_criterion=gan_class_criterion,
            device=device,
            epochs=CONFIG.gan_epochs,
            save_path=gan_checkpoint_path,
            class_names=data.class_names,
            config=CONFIG,
        )

        gan_history_path = output_dir / "gan_training_history.json"
        gan_curve_path = output_dir / "gan_training_curve.png"
        save_json(gan_history, gan_history_path)
        plot_gan_history(gan_history, gan_curve_path)

        gan_checkpoint = load_gan_checkpoint(gan, gan_checkpoint_path, device)
        print(
            "Best validation loss G : "
            f"{gan_checkpoint['best_val_generator_loss']:.6f}"
        )

    print_section("3. Generate Dataset Sintetis")
    train_embedding_stats_path = output_dir / "train_embedding_statistics.json"
    train_embedding_stats = {}
    if CONFIG.gan_per_class:
        save_skipped_analysis_note(
            train_embedding_stats_path,
            "Train embedding statistics dinonaktifkan untuk GAN per kelas karena "
            "feature space tiap discriminator tidak comparable.",
        )
    else:
        train_analysis_loader = create_analysis_loader(
            samples=data.train_samples,
            transform=data.gan_eval_transform,
            batch_size=CONFIG.batch_size,
            num_workers=CONFIG.num_workers,
        )
        train_embeddings, train_labels = encode_loader(gan, train_analysis_loader, device)
        train_embedding_stats = compute_latent_statistics(
            train_embeddings,
            train_labels,
            data.class_names,
        )
        save_latent_statistics_json(
            train_embedding_stats,
            train_embedding_stats_path,
        )

    if CONFIG.gan_per_class:
        for class_index, class_name in enumerate(data.class_names):
            class_model = gan.models[str(class_index)]
            generate_synthetic_images(
                model=class_model,
                class_names=[class_name],
                output_dir=CONFIG.synthetic_output_dir,
                images_per_class=CONFIG.generated_images_per_class,
                noise_scale=CONFIG.gan_noise_scale,
                device=device,
                seed=CONFIG.seed + class_index,
                batch_size=CONFIG.batch_size,
            )
        synthetic_samples = collect_samples_from_directory(
            root_dir=CONFIG.synthetic_output_dir,
            class_to_idx=data.class_to_idx,
        )
    else:
        synthetic_samples = generate_synthetic_images(
            model=gan,
            class_names=data.class_names,
            output_dir=CONFIG.synthetic_output_dir,
            images_per_class=CONFIG.generated_images_per_class,
            noise_scale=CONFIG.gan_noise_scale,
            device=device,
            seed=CONFIG.seed,
            batch_size=CONFIG.batch_size,
        )
    print(f"Folder gambar sintetis : {CONFIG.synthetic_output_dir}")
    print(f"Total gambar sintetis  : {len(synthetic_samples)}")

    print_section("4. Analisis Embedding dan Realism Score")
    latent_plot_path = output_dir / "embedding_distribution.png"
    latent_metrics_path = output_dir / "embedding_metrics.json"
    feature_metrics_path = output_dir / "feature_error_metrics.json"
    feature_plot_path = output_dir / "feature_error_distribution.png"
    gan_gallery_path = output_dir / "gan_sample_gallery.png"

    test_analysis_loader = create_analysis_loader(
        samples=data.test_samples,
        transform=data.gan_eval_transform,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
    )
    gan_analysis = collect_gan_analysis_batches(
        model=gan,
        data_loader=test_analysis_loader,
        device=device,
        preview_count=CONFIG.gan_preview_count,
        noise_scale=CONFIG.gan_noise_scale,
        config=CONFIG,
    )
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
        print("Analisis embedding/feature : dilewati (mode GAN per kelas)")
    else:
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
        save_latent_metrics_bundle(latent_metrics, latent_metrics_path)

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
            class_stats=train_embedding_stats,
            class_names=data.class_names,
        )
        save_feature_metrics_json(feature_metrics, feature_metrics_path)
        plot_gan_feature_distribution(
            real_scores=gan_analysis["real_scores"],
            synthetic_scores=gan_analysis["synthetic_scores"],
            synthetic_feature_distance=synthetic_feature_distance,
            output_path=feature_plot_path,
        )

    print_section("5. Training Classifier dengan Augmentasi Sintetis")
    classifier = build_model(
        model_name=CONFIG.model_name,
        num_classes=len(data.class_names),
        dropout=CONFIG.dropout,
        pretrained=CONFIG.pretrained,
        freeze_backbone=CONFIG.freeze_backbone,
        input_channels=CONFIG.image_channels,
    ).to(device)

    export_model_report(model=classifier, output_dir=classifier_assets_dir)
    classifier_graph_path = export_model_graph(
        model=classifier,
        input_size=(CONFIG.image_channels, CONFIG.image_size, CONFIG.image_size),
        output_dir=classifier_assets_dir,
        device=device,
        graph_name=f"{CONFIG.model_name}_architecture",
        depth=2,
    )
    print(f"Graph classifier      : {classifier_graph_path}")

    augmented_train_loader, synthetic_count = build_classifier_train_loader_with_synthetic(
        config=CONFIG,
        data=data,
        synthetic_root=CONFIG.synthetic_output_dir,
    )

    print(f"Sampel train asli     : {len(data.train_samples)}")
    print(f"Sampel sintetis       : {synthetic_count}")
    print(f"Total train classifier: {len(data.train_samples) + synthetic_count}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, classifier.parameters()),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )

    classifier_checkpoint_path = output_dir / CONFIG.model_filename
    history, best_val_accuracy = train_model(
        model=classifier,
        train_loader=augmented_train_loader,
        val_loader=data.clf_val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=CONFIG.epochs,
        save_path=classifier_checkpoint_path,
        class_names=data.class_names,
        config=CONFIG,
    )

    history_path = output_dir / "training_history.json"
    curve_path = output_dir / "training_curve.png"
    save_json(history, history_path)
    plot_history(history, curve_path)

    classifier_checkpoint = load_checkpoint(classifier, classifier_checkpoint_path, device)
    print(f"Best validation accuracy: {classifier_checkpoint['best_val_accuracy']:.2f}%")

    print_section("6. Evaluasi Classifier")
    evaluation_output = evaluate_classifier(
        model=classifier,
        test_loader=data.clf_test_loader,
        class_names=data.class_names,
        output_dir=output_dir,
        device=device,
    )

    print_section("7. Ringkasan Output")
    if CONFIG.gan_per_class:
        print(f"Checkpoint GAN            : {CONFIG.gan_models_dir}")
    else:
        print(f"Checkpoint GAN            : {gan_checkpoint_path}")
    print(f"History GAN               : {gan_history_path}")
    print(f"Curve GAN                 : {gan_curve_path}")
    print(f"Embedding metrics         : {latent_metrics_path}")
    print(f"Feature error metrics     : {feature_metrics_path}")
    print(f"GAN sample gallery        : {gan_gallery_path}")
    print(f"Gambar sintetis           : {CONFIG.synthetic_output_dir}")
    print(f"Checkpoint classifier     : {classifier_checkpoint_path}")
    print(f"Best validation accuracy  : {best_val_accuracy:.2f}%")
    print(f"Classifier history        : {history_path}")
    print(f"Classifier curve          : {curve_path}")
    print(f"Test report               : {evaluation_output['report_path']}")
    print(f"Confusion matrix          : {evaluation_output['cm_path']}")


if __name__ == "__main__":
    main()
