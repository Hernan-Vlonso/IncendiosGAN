"""Evaluación de robustez ante datos incompletos.

Genera curvas de FID y SSIM vs porcentaje de masking para medir
cómo degrada la calidad cuando faltan datos de condición.
"""

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.evaluation.fid import FIDCalculator
from src.evaluation.ssim import compute_condition_matched_ssim


def apply_random_masking(conditions: np.ndarray, masks: np.ndarray,
                         masking_pct: float, seed: int = 42) -> tuple:
    """Aplicar masking aleatorio a un porcentaje de los campos de condición.

    El masking opera a nivel de campo (igual que ConditionMaskingTransform
    durante el entrenamiento): las variables categóricas se enmascaran completas
    (todo el bloque one-hot) y las continuas se enmascaran individualmente.
    Esto garantiza que la evaluación de robustez replica el tipo de dato
    incompleto que el modelo vio durante training.

    Campos (7 en total):
      [0]     Temperatura  (continuo)
      [1]     Humedad      (continuo)
      [2]     Vel_Viento   (continuo)
      [3]     Sup_Total    (continuo)
      [4-13]  Exposicion   (one-hot 10 cats)
      [14-23] Dir_viento   (one-hot 10 cats)
      [24-27] Topografia   (one-hot 4 cats)

    Args:
        conditions: (N, 32) vectores de condición
        masks: (N, 32) máscaras originales
        masking_pct: fracción de los 7 campos a enmascarar [0, 1]
        seed: semilla

    Returns:
        (conditions_masked, masks_updated)
    """
    # Campos: (dim_start, dim_end, miss_flag_idx o None)
    _FIELDS = [
        (0,  1,  0),    # Temperatura  → miss_flag dim 28
        (1,  2,  1),    # Humedad      → miss_flag dim 29
        (2,  3,  2),    # Vel_Viento   → miss_flag dim 30
        (3,  4,  3),    # Sup_Total    → miss_flag dim 31
        (4,  14, None), # Exposicion
        (14, 24, None), # Dir_viento
        (24, 28, None), # Topografia
    ]
    n_fields = len(_FIELDS)
    n_to_mask = int(round(n_fields * masking_pct))

    rng = np.random.default_rng(seed)
    conditions = conditions.copy()
    masks = masks.copy()

    for i in range(len(conditions)):
        field_indices = rng.choice(n_fields, size=n_to_mask, replace=False)
        for fi in field_indices:
            start, end, flag_idx = _FIELDS[fi]
            conditions[i, start:end] = 0.0
            masks[i, start:end] = 1.0
            if flag_idx is not None:
                conditions[i, 28 + flag_idx] = 1.0
                masks[i, 28 + flag_idx] = 1.0

    return conditions, masks


@torch.no_grad()
def evaluate_robustness(
    generator,
    real_images: torch.Tensor,
    conditions: np.ndarray,
    masks: np.ndarray,
    cfg: Config,
    fid_calculator: FIDCalculator = None,
    masking_percentages: list = None,
    n_samples: int = 5000,
    device: torch.device = None,
    veg_profiles: np.ndarray = None,
) -> dict:
    """Evaluar robustez de la GAN ante datos incompletos.

    Genera imágenes con diferentes niveles de masking y mide FID/SSIM.

    Args:
        generator: modelo generador (ya en eval mode)
        real_images: tensor (N, 3, H, W) en [0, 1]
        conditions: array (N, 32)
        masks: array (N, 32)
        cfg: configuración
        fid_calculator: calculador con stats reales cacheadas
        masking_percentages: lista de fracciones [0, 0.1, ..., 0.9]
        n_samples: muestras por nivel de masking
        device: dispositivo

    Returns:
        dict con fid_curve, ssim_curve, masking_levels
    """
    if masking_percentages is None:
        masking_percentages = cfg.eval.masking_percentages
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator.eval()
    z_dim = cfg.model.z_dim
    n = min(n_samples, len(conditions))

    # Usar subset
    idx = np.random.default_rng(42).choice(len(conditions), size=n, replace=False)
    base_conditions = conditions[idx]
    base_masks = masks[idx]
    base_veg = veg_profiles[idx] if veg_profiles is not None else None

    fid_scores = []
    ssim_scores = []

    for pct in tqdm(masking_percentages, desc="Robustness evaluation"):
        # Aplicar masking
        masked_cond, masked_mask = apply_random_masking(
            base_conditions, base_masks, pct, seed=42
        )

        # Generar imágenes en batches
        cond_t = torch.from_numpy(masked_cond).to(device)
        mask_t = torch.from_numpy(masked_mask).to(device)
        veg_t = torch.from_numpy(base_veg).to(device) if base_veg is not None else None

        generated = []
        batch_size = cfg.eval.fid_batch_size
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            z = torch.randn(end - start, z_dim, device=device)
            if veg_t is not None and cfg.model.veg_dim > 0:
                veg_b = veg_t[start:end]
            else:
                veg_b = torch.zeros(end - start, 0, device=device)
            imgs = generator(z, cond_t[start:end], mask_t[start:end], veg_b)
            # Desnormalizar [-1, 1] → [0, 1]
            imgs = (imgs + 1) / 2
            generated.append(imgs.cpu())

        generated = torch.cat(generated, dim=0)

        # FID
        if fid_calculator is not None:
            fid = fid_calculator.compute_fid(generated)
            fid_scores.append(fid)
        else:
            fid_scores.append(None)

        # SSIM
        ssim_result = compute_condition_matched_ssim(
            real_images[:n], generated,
            base_conditions, masked_cond,
            k=cfg.eval.ssim_k_neighbors,
            n_samples=min(cfg.eval.ssim_n_samples, n),
        )
        ssim_scores.append(ssim_result["ssim_mean"])

    return {
        "masking_levels": masking_percentages,
        "fid_curve": fid_scores,
        "ssim_curve": ssim_scores,
    }
