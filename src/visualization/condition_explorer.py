"""Explorador de condiciones: sweep paramétrico para visualizar efecto."""

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def sweep_condition(generator, base_condition: np.ndarray,
                    base_mask: np.ndarray, dim_idx: int,
                    values: np.ndarray, z_dim: int = 128,
                    device: torch.device = None,
                    dim_name: str = None,
                    veg_profile: np.ndarray = None) -> tuple:
    """Generar imágenes variando una dimensión de condición.

    Args:
        generator: modelo generador
        base_condition: vector base (32,)
        base_mask: máscara base (32,)
        dim_idx: índice de la dimensión a variar
        values: array de valores para esa dimensión
        z_dim: dimensión del ruido latente
        device: dispositivo
        dim_name: nombre de la dimensión (para título)

    Returns:
        (images_tensor, fig)
    """
    if device is None:
        device = torch.device("cpu")

    generator.eval()
    n = len(values)

    # z fijo para ver solo el efecto de la condición
    z = torch.randn(1, z_dim, device=device).expand(n, -1)

    # Crear batch de condiciones variando una dimensión
    conditions = np.tile(base_condition, (n, 1))
    conditions[:, dim_idx] = values
    masks = np.tile(base_mask, (n, 1))

    cond_t = torch.from_numpy(conditions.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(masks.astype(np.float32)).to(device)
    veg_np = veg_profile if veg_profile is not None else np.zeros(10, dtype=np.float32)
    veg_t = torch.from_numpy(np.tile(veg_np, (n, 1)).astype(np.float32)).to(device)

    imgs = generator(z, cond_t, mask_t, veg_t)
    imgs = (imgs + 1) / 2  # [-1,1] → [0,1]

    # Visualizar
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for i, (ax, val) in enumerate(zip(axes, values)):
        ax.imshow(imgs[i].cpu().permute(1, 2, 0).numpy())
        ax.set_title(f"{val:.2f}", fontsize=10)
        ax.axis("off")

    title = f"Sweep: {dim_name or f'Dim {dim_idx}'}"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return imgs.cpu(), fig


@torch.no_grad()
def sweep_masking(generator, condition: np.ndarray, mask: np.ndarray,
                  dims_to_mask: list, z_dim: int = 128,
                  device: torch.device = None,
                  veg_profile: np.ndarray = None) -> tuple:
    """Generar imágenes enmascarando progresivamente dimensiones.

    Args:
        generator: modelo generador
        condition: vector de condición (32,)
        mask: máscara base (32,)
        dims_to_mask: lista de dimensiones a enmascarar progresivamente
        z_dim: dimensión del ruido
        device: dispositivo

    Returns:
        (images_tensor, fig)
    """
    if device is None:
        device = torch.device("cpu")

    generator.eval()
    n = len(dims_to_mask) + 1  # incluir sin masking

    z = torch.randn(1, z_dim, device=device).expand(n, -1)

    conditions = np.tile(condition, (n, 1))
    masks = np.tile(mask, (n, 1))

    # Aplicar masking progresivo
    for i, dim_group in enumerate(dims_to_mask):
        if isinstance(dim_group, int):
            dim_group = [dim_group]
        for d in dim_group:
            for j in range(i + 1, n):
                conditions[j, d] = 0.0
                masks[j, d] = 1.0

    cond_t = torch.from_numpy(conditions.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(masks.astype(np.float32)).to(device)
    veg_np = veg_profile if veg_profile is not None else np.zeros(10, dtype=np.float32)
    veg_t = torch.from_numpy(np.tile(veg_np, (n, 1)).astype(np.float32)).to(device)

    imgs = generator(z, cond_t, mask_t, veg_t)
    imgs = (imgs + 1) / 2

    # Visualizar
    labels = ["Completo"] + [f"+Mask dim {d}" for d in dims_to_mask]
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img.cpu().permute(1, 2, 0).numpy())
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    plt.suptitle("Efecto del Masking Progresivo", fontsize=14)
    plt.tight_layout()

    return imgs.cpu(), fig
