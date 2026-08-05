"""Generación de grids de vegetación y elevación usando Perlin noise."""

import numpy as np

try:
    from noise import pnoise2
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False

from src.config import CAConfig


def _simple_perlin_2d(shape, scale=0.1, octaves=4, seed=0):
    """Generación de noise 2D usando la librería noise, o fallback simple."""
    h, w = shape
    rng = np.random.default_rng(seed)

    if HAS_NOISE:
        base = rng.integers(0, 1000)
        grid = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                grid[y, x] = pnoise2(
                    x * scale + base, y * scale + base,
                    octaves=octaves,
                    persistence=0.5,
                    lacunarity=2.0,
                )
        return grid
    else:
        # Fallback: multi-octave gradient noise
        grid = np.zeros((h, w), dtype=np.float32)
        for octave in range(octaves):
            freq = 2 ** octave
            amp = 0.5 ** octave
            noise = rng.standard_normal((
                max(2, int(h * scale * freq) + 2),
                max(2, int(w * scale * freq) + 2),
            ))
            # Interpolar a tamaño del grid
            from scipy.ndimage import zoom
            factor_y = h / noise.shape[0]
            factor_x = w / noise.shape[1]
            interpolated = zoom(noise, (factor_y, factor_x), order=3)
            grid += amp * interpolated[:h, :w].astype(np.float32)
        return grid


def generate_vegetation_grid(veg_profile: np.ndarray, cfg: CAConfig,
                             seed: int = 0) -> np.ndarray:
    """Generar grid de vegetación 64x64 a partir de un perfil de proporciones.

    Usa Perlin noise para crear distribución espacial coherente.
    El noise field se cuantiza según las proporciones del perfil.

    Args:
        veg_profile: array (10,) con proporciones fraccionarias
        cfg: configuración del CA
        seed: semilla para reproducibilidad

    Returns:
        grid (64, 64) con tipo de vegetación (0-9) por celda
    """
    size = cfg.grid_size
    noise_field = _simple_perlin_2d(
        (size, size), scale=cfg.noise_scale, octaves=cfg.noise_octaves, seed=seed
    )

    # Normalizar noise a [0, 1]
    nmin, nmax = noise_field.min(), noise_field.max()
    if nmax > nmin:
        noise_field = (noise_field - nmin) / (nmax - nmin)
    else:
        noise_field = np.full_like(noise_field, 0.5)

    # Crear umbrales acumulados a partir del perfil
    profile = veg_profile.copy()
    profile = np.maximum(profile, 0)
    total = profile.sum()
    if total > 0:
        profile = profile / total
    else:
        profile = np.ones(10) / 10

    cumulative = np.cumsum(profile)
    cumulative[-1] = 1.0  # Asegurar que cubra todo

    # Garantizar monotonía estricta (tipos con proporción 0 obtienen epsilon)
    for i in range(1, len(cumulative)):
        if cumulative[i] <= cumulative[i - 1]:
            cumulative[i] = cumulative[i - 1] + 1e-10

    # Asignar tipo de vegetación según cuantiles
    flat_noise = noise_field.ravel()
    flat_veg = np.digitize(flat_noise, cumulative)
    flat_veg = np.clip(flat_veg, 0, 9)
    veg_grid = flat_veg.reshape(size, size)

    return veg_grid


def generate_elevation_grid(topography: str, cfg: CAConfig,
                            seed: int = 0) -> np.ndarray:
    """Generar grid de elevación 64x64.

    La amplitud depende del tipo de topografía.

    Args:
        topography: "Suave", "Irregular", "Abrupta" o "Missing"
        cfg: configuración del CA
        seed: semilla

    Returns:
        grid (64, 64) con valores de elevación normalizados [0, 1]
    """
    amplitude_map = {
        "Suave": 0.2,
        "Irregular": 0.5,
        "Abrupta": 1.0,
        "Missing": 0.5,  # default medio
    }
    amplitude = amplitude_map.get(topography, 0.5)

    size = cfg.grid_size
    elevation = _simple_perlin_2d(
        (size, size), scale=cfg.noise_scale * 0.5,
        octaves=cfg.noise_octaves, seed=seed + 10000
    )

    # Normalizar a [0, amplitude]
    emin, emax = elevation.min(), elevation.max()
    if emax > emin:
        elevation = (elevation - emin) / (emax - emin) * amplitude
    else:
        elevation = np.full_like(elevation, amplitude * 0.5)

    return elevation


def compute_slope(elevation: np.ndarray) -> np.ndarray:
    """Calcular pendiente (gradiente) del grid de elevación.

    Returns:
        slope (64, 64) magnitud del gradiente normalizada [0, 1]
    """
    dy, dx = np.gradient(elevation)
    slope = np.sqrt(dx**2 + dy**2)
    smax = slope.max()
    if smax > 0:
        slope = slope / smax
    return slope.astype(np.float32)


def compute_slope_direction(elevation: np.ndarray) -> np.ndarray:
    """Calcular dirección de la pendiente (ángulo en radianes).

    Returns:
        angle (64, 64) en radianes [-pi, pi]
    """
    dy, dx = np.gradient(elevation)
    return np.arctan2(dy, dx).astype(np.float32)
