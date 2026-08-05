"""Generación de heatmaps de propagación mediante Monte Carlo.

Ejecuta N simulaciones del CA con el mismo terreno y condiciones pero
diferentes semillas estocásticas. Para cada celda, la fracción de corridas
en que se quemó define su probabilidad de propagación → heatmap continuo [0, 1].
"""

import numpy as np

from src.ca.cellular_automaton import CellularAutomaton, BURNING, BURNED_OUT
from src.ca.terrain import (
    generate_vegetation_grid, generate_elevation_grid,
    compute_slope, compute_slope_direction,
)
from src.config import CAConfig


def calibrate_n_steps(sup_total_z: float, cfg: CAConfig) -> int:
    """Calibrar n_steps según el tamaño real del incendio (Superficie total Has).

    El z-score de log1p(area) almacenado en el condition_vector se mapea
    linealmente al rango [n_steps_min, n_steps] de CAConfig:
      - z <= -2  →  n_steps_min (incendios muy pequeños, <1 ha típicamente)
      - z >= +2  →  n_steps     (incendios grandes, >100 ha típicamente)

    Esto corrige la inconsistencia donde el 63% de incendios <1 ha recibían
    los mismos 80 pasos de simulación que incendios de cientos de hectáreas.
    """
    z_clamped = float(np.clip(sup_total_z, -2.0, 2.0))
    ratio = (z_clamped + 2.0) / 4.0  # [0, 1]
    n = int(round(cfg.n_steps_min + ratio * (cfg.n_steps - cfg.n_steps_min)))
    return max(cfg.n_steps_min, min(cfg.n_steps, n))


def run_monte_carlo(veg_grid: np.ndarray, elevation: np.ndarray,
                    slope: np.ndarray, slope_dir: np.ndarray,
                    wind_dir: str, wind_speed: float,
                    humidity: float, temperature: float,
                    cfg: CAConfig, n_runs: int = 30,
                    base_seed: int = 0,
                    n_horizons: int = 4,
                    sup_total_z: float = 0.0) -> list:
    """Ejecutar Monte Carlo y generar heatmaps a distintos horizontes temporales.

    Para cada horizonte temporal, acumula cuántas veces cada celda fue
    alcanzada por el fuego en N corridas, y normaliza a [0, 1].

    Args:
        veg_grid: grid de vegetación (64, 64)
        elevation: grid de elevación (64, 64)
        slope: grid de pendiente (64, 64)
        slope_dir: grid de dirección de pendiente (64, 64)
        wind_dir: dirección del viento
        wind_speed: velocidad del viento [0, 1]
        humidity: humedad relativa [0, 1]
        temperature: temperatura normalizada [0, 1]
        cfg: configuración del CA
        n_runs: número de corridas Monte Carlo
        base_seed: semilla base
        n_horizons: número de horizontes temporales a capturar
        sup_total_z: z-score de log1p(Sup_Total) para calibrar n_steps

    Returns:
        Lista de n_horizons heatmaps, cada uno (64, 64) con valores [0, 1]
    """
    size = cfg.grid_size
    n_steps = calibrate_n_steps(sup_total_z, cfg)

    # Pasos en los que capturar heatmaps
    horizon_steps = [
        int((i + 1) / n_horizons * n_steps) for i in range(n_horizons)
    ]

    # Acumuladores: cuántas veces cada celda fue alcanzada por el fuego
    burn_counts = [np.zeros((size, size), dtype=np.float32) for _ in range(n_horizons)]

    for run in range(n_runs):
        seed = base_seed + run * 7919  # primos para dispersión

        ca = CellularAutomaton(
            veg_grid=veg_grid, elevation=elevation,
            slope=slope, slope_dir=slope_dir,
            wind_dir=wind_dir, wind_speed=wind_speed,
            humidity=humidity, temperature=temperature,
            cfg=cfg,
        )
        ca.ignite()

        rng = np.random.default_rng(seed)
        horizon_idx = 0

        for step in range(n_steps):
            ca.step(rng)

            # Capturar en el horizonte correspondiente
            if horizon_idx < n_horizons and (step + 1) >= horizon_steps[horizon_idx]:
                burned = (ca.state == BURNING) | (ca.state == BURNED_OUT)
                burn_counts[horizon_idx] += burned.astype(np.float32)
                horizon_idx += 1

            # Si no hay más fuego, marcar horizontes restantes
            if not np.any(ca.state == BURNING):
                burned = (ca.state == BURNED_OUT)
                while horizon_idx < n_horizons:
                    burn_counts[horizon_idx] += burned.astype(np.float32)
                    horizon_idx += 1
                break

    # Normalizar a [0, 1]
    heatmaps = []
    for counts in burn_counts:
        heatmap = counts / n_runs
        heatmaps.append(heatmap)

    return heatmaps


def generate_heatmaps_for_record(
    veg_profile: np.ndarray,
    wind_dir: str, wind_speed: float,
    humidity: float, temperature: float,
    topography: str,
    cfg: CAConfig,
    n_runs: int = 30,
    n_horizons: int = 4,
    n_terrain_variations: int = 3,
    base_seed: int = 0,
    sup_total_z: float = 0.0,
) -> list:
    """Generar heatmaps Monte Carlo para un registro con múltiples variaciones.

    Args:
        veg_profile: perfil de vegetación (10,)
        wind_dir, wind_speed, humidity, temperature: condiciones meteorológicas
        topography: tipo de topografía
        cfg: configuración del CA
        n_runs: corridas Monte Carlo por variación
        n_horizons: horizontes temporales por variación
        n_terrain_variations: variaciones de terreno
        base_seed: semilla base
        sup_total_z: z-score de log1p(Sup_Total) para calibrar n_steps

    Returns:
        Lista de (heatmap, veg_grid) tuples.
        Total: n_terrain_variations * n_horizons heatmaps.
    """
    # Run 26: generar CA en grid 2× y recortar centro cfg.grid_size×cfg.grid_size.
    # Elimina el truncamiento en bordes que degradaba el condicionamiento direccional.
    import copy
    cfg_large = copy.copy(cfg)
    cfg_large.grid_size = cfg.grid_size * 2
    crop_size = cfg.grid_size
    margin = cfg.grid_size // 2   # = grid_large // 4 = crop_size // 2

    results = []

    for var_idx in range(n_terrain_variations):
        seed = base_seed + var_idx * 1000

        # Generar terreno en grid grande
        veg_grid_large = generate_vegetation_grid(veg_profile, cfg_large, seed=seed)
        elevation_large = generate_elevation_grid(topography, cfg_large, seed=seed)
        slope_large     = compute_slope(elevation_large)
        slope_dir_large = compute_slope_direction(elevation_large)

        # Monte Carlo en grid grande
        heatmaps_large = run_monte_carlo(
            veg_grid=veg_grid_large, elevation=elevation_large,
            slope=slope_large, slope_dir=slope_dir_large,
            wind_dir=wind_dir, wind_speed=wind_speed,
            humidity=humidity, temperature=temperature,
            cfg=cfg_large, n_runs=n_runs,
            base_seed=seed, n_horizons=n_horizons,
            sup_total_z=sup_total_z,
        )

        # Recortar centro crop_size×crop_size de cada heatmap
        # veg_grid también se recorta para mantener consistencia visual
        veg_grid_crop = veg_grid_large[margin:margin + crop_size,
                                       margin:margin + crop_size]
        for h_idx, heatmap in enumerate(heatmaps_large):
            heatmap_crop = heatmap[margin:margin + crop_size,
                                   margin:margin + crop_size]
            results.append((heatmap_crop, veg_grid_crop, var_idx, h_idx))

    return results
