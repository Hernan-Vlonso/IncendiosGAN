"""SSIM condition-matched para evaluar fidelidad condicional."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def _ssim_single(img1: torch.Tensor, img2: torch.Tensor,
                 window_size: int = 11, C1: float = 0.01**2,
                 C2: float = 0.03**2) -> float:
    """Calcular SSIM entre dos imágenes (3, H, W) en rango [0, 1]."""
    # Promediar sobre canales
    ssim_vals = []
    for c in range(img1.shape[0]):
        x = img1[c].unsqueeze(0).unsqueeze(0)
        y = img2[c].unsqueeze(0).unsqueeze(0)

        # Ventana gaussiana
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = g.outer(g)
        window = window / window.sum()
        window = window.unsqueeze(0).unsqueeze(0).to(x.device)

        mu_x = F.conv2d(x, window, padding=window_size // 2)
        mu_y = F.conv2d(y, window, padding=window_size // 2)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size // 2) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, window, padding=window_size // 2) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=window_size // 2) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = num / den
        ssim_vals.append(ssim_map.mean().item())

    return np.mean(ssim_vals)


@torch.no_grad()
def compute_condition_matched_ssim(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    real_conditions: np.ndarray,
    gen_conditions: np.ndarray,
    k: int = 5,
    n_samples: int = 1000,
) -> dict:
    """Calcular SSIM entre imágenes generadas y sus vecinos reales más cercanos.

    Para cada imagen generada, encuentra las k imágenes reales con condiciones
    más similares (coseno) y calcula SSIM promedio.

    Args:
        real_images: (N_real, 3, H, W) en [0, 1]
        generated_images: (N_gen, 3, H, W) en [0, 1]
        real_conditions: (N_real, D) vectores de condición
        gen_conditions: (N_gen, D) vectores de condición
        k: número de vecinos
        n_samples: número de muestras generadas a evaluar

    Returns:
        dict con ssim_mean, ssim_std, ssim_per_sample
    """
    n_gen = min(n_samples, len(generated_images))

    # Similitud coseno entre condiciones
    n_real = len(real_images)
    sim_matrix = cosine_similarity(gen_conditions[:n_gen], real_conditions[:n_real])

    ssim_scores = []
    for i in range(n_gen):
        # Top-k vecinos más similares
        top_k_idx = np.argsort(sim_matrix[i])[-k:]

        # SSIM promedio con los k vecinos
        gen_img = generated_images[i]
        neighbor_ssims = []
        for j in top_k_idx:
            real_img = real_images[j]
            s = _ssim_single(gen_img, real_img)
            neighbor_ssims.append(s)
        ssim_scores.append(np.mean(neighbor_ssims))

    ssim_scores = np.array(ssim_scores)
    return {
        "ssim_mean": float(ssim_scores.mean()),
        "ssim_std": float(ssim_scores.std()),
        "ssim_per_sample": ssim_scores,
    }
