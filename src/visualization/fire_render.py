"""Post-procesado de la salida cruda del generador para visualizar propagación."""

import numpy as np
from scipy.ndimage import gaussian_filter

# Dirección downwind esperada en coordenadas de imagen (dx = columna, dy = fila; fila 0 = Norte)
DOWNWIND = {
    "N": (0, 1), "NE": (-1, 1), "E": (-1, 0), "SE": (-1, -1),
    "S": (0, -1), "SW": (1, -1), "W": (1, 0), "NW": (1, 1),
}


def fire_intensity(img_rgb: np.ndarray, smooth: float = 1.2) -> np.ndarray:
    """Canal escalar de intensidad de fuego con remoción del fondo del generador."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    raw = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
    sm = gaussian_filter(raw, sigma=smooth)
    return np.maximum(sm - np.percentile(sm, 8), 0)


def centroid_offset(fire: np.ndarray):
    """Desplazamiento (dx, dy) en píxeles del centroide de fuego respecto del centro."""
    total = fire.sum()
    if total < 1e-6:
        return None
    ys, xs = np.mgrid[0:fire.shape[0], 0:fire.shape[1]]
    return (fire * xs).sum() / total - fire.shape[1] / 2, (fire * ys).sum() / total - fire.shape[0] / 2


def direction_error_deg(fire: np.ndarray, wind_dir: str):
    """Ángulo entre el desplazamiento del centroide y la dirección downwind esperada."""
    offset = centroid_offset(fire)
    if offset is None or wind_dir not in DOWNWIND:
        return None
    d = np.array(offset)
    norm = np.linalg.norm(d)
    if norm < 2:
        return 180.0
    e = np.array(DOWNWIND[wind_dir], dtype=float)
    return float(np.degrees(np.arccos(np.clip(d @ e / (norm * np.linalg.norm(e)), -1, 1))))
