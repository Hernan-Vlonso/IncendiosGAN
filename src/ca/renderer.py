"""Renderizado de estados del CA a imágenes RGB 64x64."""

import numpy as np
from PIL import Image

from src.ca.cellular_automaton import UNBURNED, BURNING, BURNED_OUT, NON_FLAMMABLE


# Colores base por tipo de vegetación (RGB, escala 0-255)
VEGETATION_COLORS = {
    0: (34, 139, 34),    # Arbolado_Nat — verde bosque
    1: (0, 100, 0),      # Arbolado_Exot — verde oscuro
    2: (85, 170, 85),    # Arbolado_Mixto — verde medio
    3: (154, 205, 50),   # Matorral — verde amarillento
    4: (210, 225, 120),  # Pastizal — verde claro/amarillo
    5: (218, 165, 32),   # Agricola — dorado
    6: (139, 119, 101),  # Desechos — marrón claro
    7: (107, 142, 35),   # Plantacion_Joven — verde oliva
    8: (169, 169, 169),  # Sin_Vegetacion — gris
    9: (60, 120, 60),    # Otro_Comb — verde tenue
}

# Color para fuego activo (gradiente naranja-rojo)
FIRE_COLORS = [
    (255, 200, 0),    # inicio del fuego — amarillo
    (255, 140, 0),    # fuego medio — naranja
    (255, 69, 0),     # fuego intenso — rojo-naranja
    (200, 0, 0),      # fuego tardío — rojo oscuro
]

# Color para zona quemada
BURNED_COLOR = (40, 40, 40)  # casi negro


def render_state(ca_state: np.ndarray, veg_grid: np.ndarray,
                 step: int = 0, total_steps: int = 80) -> np.ndarray:
    """Renderizar un estado del CA a imagen RGB.

    Args:
        ca_state: grid (64, 64) con estados del CA
        veg_grid: grid (64, 64) con tipos de vegetación
        step: paso actual de la simulación
        total_steps: total de pasos

    Returns:
        imagen RGB (64, 64, 3) como uint8
    """
    h, w = ca_state.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Capa base: vegetación
    for veg_type, color in VEGETATION_COLORS.items():
        mask = (veg_grid == veg_type) & (ca_state == UNBURNED)
        img[mask] = color

    # No inflamable
    mask_nf = ca_state == NON_FLAMMABLE
    img[mask_nf] = VEGETATION_COLORS[8]

    # Fuego activo — color según progresión temporal
    burning_mask = ca_state == BURNING
    if burning_mask.any():
        progress = min(step / max(total_steps, 1), 1.0)
        fire_idx = min(int(progress * (len(FIRE_COLORS) - 1)), len(FIRE_COLORS) - 1)
        fire_color = FIRE_COLORS[fire_idx]
        img[burning_mask] = fire_color

    # Zona quemada
    burned_mask = ca_state == BURNED_OUT
    img[burned_mask] = BURNED_COLOR

    return img


def render_to_pil(ca_state: np.ndarray, veg_grid: np.ndarray,
                  step: int = 0, total_steps: int = 80) -> Image.Image:
    """Renderizar estado a imagen PIL."""
    rgb = render_state(ca_state, veg_grid, step, total_steps)
    return Image.fromarray(rgb, mode="RGB")


def render_heatmap(heatmap: np.ndarray, veg_grid: np.ndarray) -> np.ndarray:
    """Renderizar un heatmap de probabilidad de propagación a imagen RGB.

    Combina la capa base de vegetación con un overlay del heatmap de fuego.
    Zonas con probabilidad 0 muestran vegetación original.
    Zonas con probabilidad alta muestran gradiente negro→rojo→naranja→amarillo→blanco.

    Args:
        heatmap: array (64, 64) con probabilidades [0, 1]
        veg_grid: array (64, 64) con tipos de vegetación

    Returns:
        imagen RGB (64, 64, 3) como uint8
    """
    h, w = heatmap.shape
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Capa base: vegetación (atenuada)
    for veg_type, color in VEGETATION_COLORS.items():
        mask = veg_grid == veg_type
        img[mask] = np.array(color, dtype=np.float32)

    # Colormap de fuego: negro → rojo → naranja → amarillo → blanco
    fire_cmap = np.array([
        [0, 0, 0],         # 0.0 — sin fuego (transparente)
        [80, 0, 0],        # 0.1 — rojo muy oscuro
        [160, 20, 0],      # 0.2 — rojo oscuro
        [200, 50, 0],      # 0.3 — rojo
        [220, 100, 0],     # 0.4 — rojo-naranja
        [240, 140, 0],     # 0.5 — naranja
        [255, 180, 0],     # 0.6 — naranja-amarillo
        [255, 220, 50],    # 0.7 — amarillo
        [255, 240, 150],   # 0.8 — amarillo claro
        [255, 250, 220],   # 0.9 — casi blanco
        [255, 255, 255],   # 1.0 — blanco (máxima probabilidad)
    ], dtype=np.float32)

    # Interpolar color del heatmap
    prob = heatmap.ravel()
    indices = prob * (len(fire_cmap) - 1)
    idx_low = np.floor(indices).astype(int)
    idx_high = np.minimum(idx_low + 1, len(fire_cmap) - 1)
    frac = (indices - idx_low)[:, None]

    fire_colors = (1 - frac) * fire_cmap[idx_low] + frac * fire_cmap[idx_high]
    fire_colors = fire_colors.reshape(h, w, 3)

    # Blending: donde hay probabilidad > 0, mezclar vegetación con fuego
    alpha = heatmap[:, :, None]  # (h, w, 1)
    img = (1 - alpha) * img + alpha * fire_colors

    return np.clip(img, 0, 255).astype(np.uint8)


def render_heatmap_to_pil(heatmap: np.ndarray,
                          veg_grid: np.ndarray) -> Image.Image:
    """Renderizar heatmap a imagen PIL."""
    rgb = render_heatmap(heatmap, veg_grid)
    return Image.fromarray(rgb, mode="RGB")


def render_sequence(snapshots: list, veg_grid: np.ndarray,
                    total_steps: int = 80) -> list:
    """Renderizar una secuencia de snapshots a imágenes PIL.

    Args:
        snapshots: lista de arrays (64, 64) con estados del CA
        veg_grid: grid de vegetación
        total_steps: total de pasos de simulación

    Returns:
        Lista de imágenes PIL
    """
    n = len(snapshots)
    images = []
    for i, state in enumerate(snapshots):
        step = int((i + 1) / n * total_steps)
        img = render_to_pil(state, veg_grid, step, total_steps)
        images.append(img)
    return images
