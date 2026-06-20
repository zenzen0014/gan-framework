from pathlib import Path

import torch
import torch.nn as nn
from torchview import draw_graph


def export_model_graph(
    model: nn.Module,
    input_size: tuple[int, int, int] | None,
    output_dir: str | Path,
    device: torch.device,
    graph_name: str = "model_architecture",
    depth: int = 2,
    input_data=None,
) -> Path:
    """
    Export arsitektur model sebagai gambar PNG menggunakan torchview.

    Parameters
    ----------
    model:
        Model PyTorch yang akan divisualisasikan.

    input_size:
        Ukuran input gambar dalam format (C, H, W), misalnya (3, 224, 224).

    output_dir:
        Folder output untuk menyimpan gambar.

    device:
        Device yang digunakan, CPU atau CUDA.

    graph_name:
        Nama file output tanpa ekstensi.

    depth:
        Kedalaman detail module.
        depth=1 lebih ringkas.
        depth=2 menampilkan sub-module.
        depth=3 lebih detail, tetapi bisa sangat besar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    sample_input = input_data
    if sample_input is None:
        if input_size is None:
            raise ValueError("input_size atau input_data harus diberikan.")

        sample_input = torch.zeros(
            1,
            input_size[0],
            input_size[1],
            input_size[2],
            device=device,
        )

    model_graph = draw_graph(
        model,
        input_data=sample_input,
        graph_name=graph_name,
        expand_nested=True,
        depth=depth,
        device=str(device),
        save_graph=True,
        directory=str(output_dir),
        filename=graph_name,
    )

    # torchview biasanya menyimpan sebagai .png jika format default tersedia
    output_path = output_dir / f"{graph_name}.png"

    return output_path
