"""Data augmentation: transforms de imagen y condition masking."""

import numpy as np
import torch
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Augmentación rotacional (Run 29)
# ---------------------------------------------------------------------------

# Shift del índice direccional para cada k de torch.rot90 (rotación en dims=[1,2]):
#   k=1 → 90° CCW en imagen → fire going N aparece a la derecha (E) → shift +2
#   k=2 → 180°                                                       → shift +4
#   k=3 → 90° CW  en imagen → fire going N aparece a la izquierda (W)→ shift +6
_ROT_SHIFT = {0: 0, 1: 2, 2: 4, 3: 6}

# Inicio de cada campo direccional en el condition vector.
# Las 8 categorías físicas (N,NE,E,SE,S,SW,W,NW) ocupan [start:start+8].
# Las categorías no-direccionales (Plano/Missing, Calma/Missing) se dejan intactas.
_DIR_FIELD_STARTS = [4, 14]  # Exposicion: [4-11], Dir_viento: [14-21]


def get_image_transforms(img_size: int = 64, is_train: bool = True) -> T.Compose:
    """Transforms para imágenes del CA.

    Las imágenes del CA ya son 64x64, pero aplicamos augmentation geométrico
    durante entrenamiento.
    """
    if is_train:
        # Flips eliminados (bug P1, run14): corrompen la señal direccional.
        # Reemplazo de regularización con augmentación direction-preserving:
        #   - RandomCrop(padding=4): jitter espacial pequeño sin rotación
        #   - ColorJitter: variación de intensidad que simula distintas
        #     condiciones de vegetación y resolución del heatmap
        return T.Compose([
            T.Resize(img_size),
            T.RandomCrop(img_size, padding=4, padding_mode='reflect'),
            T.ColorJitter(brightness=0.15, contrast=0.10),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])
    else:
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])


def apply_condition_masking(condition: np.ndarray, mask: np.ndarray,
                            mask_prob: float = 0.3,
                            rng: np.random.Generator = None) -> tuple:
    """Aplicar condition masking como data augmentation.

    Enmascara a nivel de campo (no de dimensión individual) para respetar
    la estructura one-hot de las variables categóricas. Un campo categórico
    se enmascara completo (todo el bloque one-hot a 0) o no se enmascara.

    Layout del vector (32 dims):
      [0]     Temperatura     (continuo)
      [1]     Humedad         (continuo)
      [2]     Vel_Viento      (continuo)
      [3]     Sup_Total       (continuo)
      [4-13]  Exposicion      (one-hot 10 cats)
      [14-23] Dir_viento      (one-hot 10 cats)
      [24-27] Topografia      (one-hot 4 cats)
      [28-31] missingness flags (no se enmascaran)

    Args:
        condition: vector de condición (32,)
        mask: máscara de missingness original (32,)
        mask_prob: probabilidad de masking por campo
        rng: generador de números aleatorios

    Returns:
        (condition_masked, mask_updated)
    """
    if rng is None:
        rng = np.random.default_rng()

    condition = condition.copy()
    mask = mask.copy()

    # Campos: (dim_start, dim_end, cont_flag_idx_or_None)
    # cont_flag_idx: índice del miss_flag en dims 28-31 (solo continuos)
    fields = [
        (0,  1,  0),    # Temperatura  → miss_flag dim 28
        (1,  2,  1),    # Humedad      → miss_flag dim 29
        (2,  3,  2),    # Vel_Viento   → miss_flag dim 30
        (3,  4,  3),    # Sup_Total    → miss_flag dim 31
        (4,  14, None), # Exposicion   (one-hot completo)
        (14, 24, None), # Dir_viento   (one-hot completo)
        (24, 28, None), # Topografia   (one-hot completo)
    ]

    for start, end, flag_idx in fields:
        if rng.random() < mask_prob:
            condition[start:end] = 0.0
            mask[start:end] = 1.0
            if flag_idx is not None:
                condition[28 + flag_idx] = 1.0
                mask[28 + flag_idx] = 1.0

    return condition, mask


class ConditionMaskingTransform:
    """Transform callable para condition masking en el DataLoader."""

    def __init__(self, mask_prob: float = 0.3, seed: int = None):
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, condition: np.ndarray, mask: np.ndarray) -> tuple:
        return apply_condition_masking(condition, mask, self.mask_prob, self.rng)


class RotationalAugmentTransform:
    """Augmentación rotacional discreta 0/90/180/270° con corrección de labels.

    Rota la imagen del heatmap (tensor CHW) y actualiza simultáneamente los
    campos direccionales del condition vector (Exposicion y Dir_viento) para
    que imagen y condición sigan siendo consistentes.

    Rotación y shift de dirección son inseparables: nunca se aplica uno sin
    el otro. Solo usar en entrenamiento.

    Corrección de label:
      Imagen rotada 90° CCW (k=1): N→E, E→S, S→W, W→N, NE→SE, ...
      Imagen rotada 180°    (k=2): N→S, E→W, S→N, W→E, NE→SW, ...
      Imagen rotada 90° CW  (k=3): N→W, W→S, S→E, E→N, NE→NW, ...
    Las categorías no-direccionales (Plano, Missing, Calma) no se modifican.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: torch.Tensor,
                 condition: np.ndarray) -> tuple:
        """
        Args:
            img:       tensor (C, H, W) normalizado [-1, 1]
            condition: array  (32,) vector de condición

        Returns:
            (img_rotated, condition_rotated)
        """
        k = int(self.rng.integers(0, 4))  # 0, 1, 2, 3
        if k == 0:
            return img, condition

        # Rotar imagen
        img = torch.rot90(img, k, dims=[1, 2])

        # Rotar campos direccionales del condition vector
        shift = _ROT_SHIFT[k]
        condition = condition.copy()
        for start in _DIR_FIELD_STARTS:
            block = condition[start:start + 8].copy()
            condition[start:start + 8] = np.roll(block, shift)

        return img, condition
