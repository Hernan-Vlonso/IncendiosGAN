"""DiffAugment: Differentiable Augmentation for Data-Efficient GAN Training.

Implementación basada en Zhao et al. 2020.
Aplica augmentaciones diferenciables (color, translation, cutout) a imágenes,
permitiendo que los gradientes fluyan a través de las transformaciones.
"""

import torch
import torch.nn.functional as F


def DiffAugment(x, policy="color,translation,cutout"):
    """Aplicar augmentaciones diferenciables a un batch de imágenes.

    Args:
        x: tensor (B, C, H, W)
        policy: string con augmentaciones separadas por coma.
                Opciones: "color", "translation", "cutout"

    Returns:
        tensor augmentado con misma shape que x
    """
    if policy == "" or policy is None:
        return x
    for p in policy.split(","):
        p = p.strip()
        if p in AUGMENT_FNS:
            for fn in AUGMENT_FNS[p]:
                x = fn(x)
    return x


def _rand_brightness(x):
    """Jitter de brillo diferenciable."""
    return torch.clamp(x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5), -1, 1)


def _rand_saturation(x):
    """Jitter de saturación diferenciable."""
    x_mean = x.mean(dim=1, keepdim=True)
    alpha = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    return alpha * x + (1 - alpha) * x_mean


def _rand_contrast(x):
    """Jitter de contraste diferenciable."""
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    alpha = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    return alpha * x + (1 - alpha) * x_mean


def _rand_translation(x, ratio=0.125):
    """Traslación aleatoria diferenciable con padding."""
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1],
                                  device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1],
                                  device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[
        grid_batch, grid_x, grid_y
    ].permute(0, 3, 1, 2).contiguous()
    return x


def _rand_cutout(x, ratio=0.5):
    """Cutout aleatorio diferenciable (rectángulo a cero)."""
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2),
                              size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2),
                              size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    "color": [_rand_brightness, _rand_saturation, _rand_contrast],
    "translation": [_rand_translation],
    "cutout": [_rand_cutout],
}
