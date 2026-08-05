"""Grids comparativos: imágenes reales vs generadas."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


def show_image_grid(images: torch.Tensor, nrow: int = 8,
                    title: str = "", save_path: str = None):
    """Mostrar grid de imágenes.

    Args:
        images: tensor (N, 3, H, W) en rango [0, 1] o [-1, 1]
        nrow: imágenes por fila
        title: título
        save_path: ruta para guardar
    """
    # Normalizar a [0, 1] si necesario
    if images.min() < 0:
        images = (images + 1) / 2

    grid = vutils.make_grid(images, nrow=nrow, normalize=False, padding=2)
    np_grid = grid.cpu().numpy().transpose(1, 2, 0)

    fig, ax = plt.subplots(figsize=(nrow * 1.5, (len(images) // nrow + 1) * 1.5))
    ax.imshow(np_grid)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def compare_real_vs_generated(real_images: torch.Tensor,
                              generated_images: torch.Tensor,
                              n_show: int = 8,
                              save_path: str = None):
    """Comparar imágenes reales y generadas lado a lado.

    Args:
        real_images: tensor (N, 3, H, W)
        generated_images: tensor (N, 3, H, W)
        n_show: número de pares
        save_path: ruta para guardar
    """
    n = min(n_show, len(real_images), len(generated_images))

    # Normalizar a [0, 1]
    real = real_images[:n]
    gen = generated_images[:n]
    if real.min() < 0:
        real = (real + 1) / 2
    if gen.min() < 0:
        gen = (gen + 1) / 2

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(real[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Real", fontsize=12)

        axes[1, i].imshow(gen[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Generado", fontsize=12)

    plt.suptitle("Real vs Generado", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def show_training_progression(sample_dir: str, epochs: list = None,
                              save_path: str = None):
    """Mostrar progresión del entrenamiento con samples de diferentes epochs."""
    from pathlib import Path
    from PIL import Image
    import torchvision.transforms as T

    sample_dir = Path(sample_dir)
    files = sorted(sample_dir.glob("samples_epoch*.png"))

    if not files:
        print("No se encontraron samples.")
        return None

    if epochs is not None:
        selected = []
        for f in files:
            ep = int(f.stem.split("epoch")[1])
            if ep in epochs:
                selected.append(f)
        files = selected

    # Limitar a 6 imágenes
    if len(files) > 6:
        step = len(files) // 6
        files = files[::step][:6]

    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 4))
    if len(files) == 1:
        axes = [axes]

    for ax, f in zip(axes, files):
        img = Image.open(f)
        ax.imshow(np.array(img))
        epoch = f.stem.split("epoch")[1]
        ax.set_title(f"Epoch {epoch}")
        ax.axis("off")

    plt.suptitle("Progresión del Entrenamiento", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
