from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


plt.rcParams["font.family"] = "DejaVu Sans"


def count_model_parameters(model: nn.Module) -> pd.DataFrame:
    """
    Membuat tabel parameter model per layer/module.
    Tabel ini membantu melihat bagian mana yang trainable dan frozen.
    """
    rows = []

    for name, parameter in model.named_parameters():
        rows.append(
            {
                "Layer Name": name,
                "Shape": list(parameter.shape),
                "Parameters": parameter.numel(),
                "Trainable": parameter.requires_grad,
            }
        )

    return pd.DataFrame(rows)


def summarize_parameters(parameter_table: pd.DataFrame) -> dict:
    """
    Menghitung ringkasan total parameter dan trainable parameter.
    """
    total_params = int(parameter_table["Parameters"].sum())

    trainable_params = int(
        parameter_table.loc[
            parameter_table["Trainable"] == True,
            "Parameters",
        ].sum()
    )

    frozen_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percentage": trainable_params / total_params * 100
        if total_params > 0
        else 0,
    }


def print_parameter_table(parameter_table: pd.DataFrame) -> None:
    """
    Menampilkan tabel parameter di terminal.
    """
    display_table = parameter_table.copy()
    display_table["Parameters"] = display_table["Parameters"].map("{:,}".format)
    display_table["Trainable"] = display_table["Trainable"].map(
        lambda value: "Yes" if value else "No"
    )

    print(display_table.to_string(index=False))


def save_parameter_table(
    parameter_table: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Menyimpan tabel parameter ke CSV dan PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "trainable_parameter_table.csv"
    png_path = output_dir / "trainable_parameter_table.png"

    parameter_table.to_csv(csv_path, index=False)

    display_table = parameter_table.copy()
    display_table["Parameters"] = display_table["Parameters"].map("{:,}".format)
    display_table["Trainable"] = display_table["Trainable"].map(
        lambda value: "Yes" if value else "No"
    )

    # Batasi jumlah baris agar gambar tetap terbaca
    max_rows = 30
    if len(display_table) > max_rows:
        display_table = display_table.head(max_rows)
        note = f"Showing first {max_rows} layers only. Full table saved in CSV."
    else:
        note = "Full parameter table."

    fig_height = max(4, len(display_table) * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#BFBFBF")
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_facecolor("#D9EAD3")
            cell.set_text_props(weight="bold", color="black")
        else:
            cell.set_facecolor("#FFFFFF")
            cell.set_text_props(color="black")

    ax.set_title(
        "Trainable Parameter Table",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color="black",
    )

    fig.text(
        0.01,
        0.01,
        note,
        fontsize=10,
        color="black",
        family="DejaVu Sans",
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Parameter table CSV : {csv_path}")
    print(f"Parameter table PNG : {png_path}")


def save_architecture_text(
    model: nn.Module,
    output_dir: str | Path,
) -> None:
    """
    Menyimpan arsitektur model dalam format teks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    architecture_path = output_dir / "model_architecture.txt"

    with architecture_path.open("w", encoding="utf-8") as file:
        file.write(str(model))

    print(f"Architecture TXT    : {architecture_path}")


def save_architecture_image(
    model: nn.Module,
    output_dir: str | Path,
) -> None:
    """
    Menyimpan arsitektur model sebagai gambar sederhana berbasis teks.
    Cocok untuk dokumentasi awal tanpa library tambahan seperti graphviz.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "model_architecture.png"

    architecture_text = str(model)
    lines = architecture_text.splitlines()

    max_lines = 80
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("...")
        lines.append("Architecture is truncated. Full version is saved in model_architecture.txt.")

    fig_height = max(6, len(lines) * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
        color="black",
    )

    ax.set_title(
        "Model Architecture",
        fontsize=16,
        fontweight="bold",
        family="DejaVu Sans",
        pad=20,
        color="black",
    )

    fig.tight_layout()
    fig.savefig(image_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Architecture PNG    : {image_path}")


def export_model_report(
    model: nn.Module,
    output_dir: str | Path,
) -> None:
    """
    Export lengkap:
    - tabel parameter
    - ringkasan total/trainable/frozen parameter
    - arsitektur model dalam TXT
    - arsitektur model dalam PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_table = count_model_parameters(model)
    summary = summarize_parameters(parameter_table)

    print("\nTrainable Parameter Table")
    print("=" * 80)
    print_parameter_table(parameter_table)

    print("\nParameter Summary")
    print("=" * 80)
    print(f"Total parameter     : {summary['total_params']:,}")
    print(f"Trainable parameter : {summary['trainable_params']:,}")
    print(f"Frozen parameter    : {summary['frozen_params']:,}")
    print(f"Trainable ratio     : {summary['trainable_percentage']:.2f}%")

    save_parameter_table(parameter_table, output_dir)
    save_architecture_text(model, output_dir)
    save_architecture_image(model, output_dir)