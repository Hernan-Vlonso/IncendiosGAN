"""Utilidades para visualizar vegetación como fondo de heatmaps de fuego.

El vector de vegetación v ∈ ℝ¹⁰ contiene proporciones de 10 tipos:
  0 Arbolado nativo · 1 Eucalipto · 2 Arbolado mixto · 3 Matorral
  4 Pastizal · 5 Agrícola · 6 Desechos · 7 Plantación joven
  8 Sin vegetación · 9 Pino maduro
"""
import numpy as np
import matplotlib.pyplot as plt

VEG_NAMES = [
    "Arbolado nativo", "Eucalipto", "Arbolado mixto", "Matorral",
    "Pastizal", "Agrícola", "Desechos", "Plantación joven",
    "Sin vegetación", "Pino maduro",
]

# Colores RGB [0,1]: espectro verde–marrón, contrasta con fuego naranja/amarillo
VEG_COLORS = np.array([
    [0.176, 0.416, 0.118],   # 0 Arbolado nativo    — verde oscuro
    [0.290, 0.549, 0.165],   # 1 Eucalipto           — verde medio
    [0.420, 0.490, 0.141],   # 2 Arbolado mixto      — oliva
    [0.475, 0.439, 0.125],   # 3 Matorral            — ocre verdoso
    [0.545, 0.486, 0.125],   # 4 Pastizal            — paja seca
    [0.478, 0.361, 0.188],   # 5 Agrícola            — tierra
    [0.353, 0.231, 0.125],   # 6 Desechos            — marrón oscuro
    [0.118, 0.420, 0.314],   # 7 Plantación joven    — verde azulado
    [0.353, 0.353, 0.353],   # 8 Sin vegetación      — gris
    [0.118, 0.290, 0.196],   # 9 Pino maduro         — verde muy oscuro
], dtype=np.float32)


def make_veg_background(veg_profile: np.ndarray, shape_hw: tuple) -> np.ndarray:
    """Imagen de fondo RGB a partir del perfil de vegetación (mezcla ponderada).

    Args:
        veg_profile: (10,) proporciones de cada tipo
        shape_hw:    (H, W) del grid de salida
    Returns:
        (H, W, 3) float32 en [0, 1]
    """
    prof = np.asarray(veg_profile, dtype=np.float32).copy()
    total = prof.sum()
    if total > 1e-6:
        prof /= total
    bg_color = np.clip((prof[:, None] * VEG_COLORS).sum(axis=0), 0, 1)
    H, W = shape_hw
    return np.broadcast_to(bg_color[None, None, :], (H, W, 3)).copy()


def composite_fire_on_veg(
    fire_norm: np.ndarray,
    veg_profile: np.ndarray,
    cmap: str = "inferno",
    max_fire_alpha: float = 0.90,
) -> np.ndarray:
    """Compone heatmap de fuego sobre fondo de vegetación.

    Args:
        fire_norm:      (H, W) en [0, 1] — intensidad de fuego normalizada
        veg_profile:    (10,) proporciones de vegetación
        cmap:           colormap de matplotlib para el fuego
        max_fire_alpha: opacidad máxima del fuego (el fondo siempre asoma un poco)
    Returns:
        (H, W, 3) float32 en [0, 1]
    """
    H, W = fire_norm.shape
    bg = make_veg_background(veg_profile, (H, W))

    cm = plt.get_cmap(cmap)
    fire_rgb = cm(fire_norm)[:, :, :3].astype(np.float32)

    # Alpha progresivo: baja intensidad → más vegetación visible; alta → más fuego
    alpha = np.clip(fire_norm ** 0.55 * max_fire_alpha, 0, max_fire_alpha)[..., None]
    return np.clip(bg * (1.0 - alpha) + fire_rgb * alpha, 0, 1)


def veg_legend_patches(veg_profile: np.ndarray, top_n: int = 3):
    """Devuelve patches de matplotlib para los top_n tipos predominantes."""
    import matplotlib.patches as mpatches
    prof = np.asarray(veg_profile, dtype=np.float32)
    total = prof.sum()
    if total > 1e-6:
        prof = prof / total
    top_idx = np.argsort(prof)[::-1][:top_n]
    patches = []
    for i in top_idx:
        if prof[i] < 0.03:
            break
        color = VEG_COLORS[i]
        label = f"{VEG_NAMES[i]} {prof[i]*100:.0f}%"
        patches.append(mpatches.Patch(color=color, label=label))
    return patches
