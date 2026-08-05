#!/usr/bin/env python3
"""Diagnóstico 2: Test unitario del CA por dirección.

Corre el CA directamente con cada dirección de viento, terreno plano,
vegetación uniforme, viento fuerte. Mide el centroide del área quemada.
Si el CA es correcto, el centroide debe apuntar downwind para cada dirección.
"""

import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.ca.cellular_automaton import CellularAutomaton, BURNING, BURNED_OUT
from src.ca.heatmap import run_monte_carlo

_SQRT2 = math.sqrt(2) / 2.0
_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Vectores downwind esperados en coords imagen (y hacia abajo = Sur)
# (tx, ty): tx = col direction (+1=right=East), ty = row direction (+1=down=South)
_DIR_TARGET = [
    ( 0.0,   +1.0  ),  # N   → fuego va al Sur (abajo)
    (-_SQRT2,+_SQRT2), # NE  → fuego va al SW (izq-abajo)
    (-1.0,    0.0  ),  # E   → fuego va al Oeste (izquierda)
    (-_SQRT2,-_SQRT2), # SE  → fuego va al NW (izq-arriba)
    ( 0.0,   -1.0  ),  # S   → fuego va al Norte (arriba)
    (+_SQRT2,-_SQRT2), # SW  → fuego va al NE (der-arriba)
    (+1.0,    0.0  ),  # W   → fuego va al Este (derecha)
    (+_SQRT2,+_SQRT2), # NW  → fuego va al SE (der-abajo)
]


def centroid_of_heatmap(heatmap):
    """Calcula centroide (cx, cy) del heatmap. cx=col, cy=row, origen=centro."""
    H, W = heatmap.shape
    yy = np.arange(H, dtype=np.float32) - (H - 1) / 2.0
    xx = np.arange(W, dtype=np.float32) - (W - 1) / 2.0
    mass = heatmap.sum()
    if mass < 1e-6:
        return 0.0, 0.0
    cy = (heatmap * yy[:, None]).sum() / mass
    cx = (heatmap * xx[None, :]).sum() / mass
    return cx, cy


def angular_error(cx, cy, tx, ty):
    """Error angular entre vector (cx,cy) y target (tx,ty)."""
    if abs(cx) < 1e-6 and abs(cy) < 1e-6:
        return 180.0  # sin señal = peor caso
    dot = cx * tx + cy * ty
    norm_v = math.sqrt(cx**2 + cy**2)
    norm_t = math.sqrt(tx**2 + ty**2)
    cos_a = max(-1.0, min(1.0, dot / (norm_v * norm_t)))
    return math.degrees(math.acos(cos_a))


def main():
    cfg = get_config()

    # Terreno: vegetación uniforme (Matorral=3, flammability=0.7), sin elevación
    size = cfg.ca.grid_size
    veg_grid = np.full((size, size), 3, dtype=np.int32)  # Matorral uniforme
    elevation = np.zeros((size, size), dtype=np.float32)
    slope = np.zeros((size, size), dtype=np.float32)
    slope_dir = np.zeros((size, size), dtype=np.float32)

    # Condiciones fuertes para maximizar señal de viento
    wind_speed = 1.0   # máximo (normalizado)
    humidity   = 0.3   # baja = favorece propagación
    temperature = 0.7  # alta = favorece propagación

    n_runs = 50  # MC runs por dirección

    print(f"\nTest unitario CA — terreno plano, viento máximo (wind_speed={wind_speed})")
    print(f"Grid {size}×{size}, Matorral uniforme, {n_runs} MC runs por dirección")
    print(f"\n{'Dir':>4} {'cx':>7} {'cy':>7} {'Angulo med':>12} {'Esperado':>10} {'Error':>8} {'OK?':>6}")
    print("-" * 60)

    for d, wind_dir in enumerate(_DIRS):
        heatmaps = run_monte_carlo(
            veg_grid=veg_grid, elevation=elevation,
            slope=slope, slope_dir=slope_dir,
            wind_dir=wind_dir, wind_speed=wind_speed,
            humidity=humidity, temperature=temperature,
            cfg=cfg.ca, n_runs=n_runs,
            base_seed=d * 1000, n_horizons=4,
            sup_total_z=0.0,
        )

        # Usar el último horizonte (máxima propagación)
        heatmap = heatmaps[-1]

        cx, cy = centroid_of_heatmap(heatmap)
        angle_measured = math.degrees(math.atan2(cy, cx))

        tx, ty = _DIR_TARGET[d]
        angle_expected = math.degrees(math.atan2(ty, tx))
        err = angular_error(cx, cy, tx, ty)
        ok = "OK" if err < 45 else "FAIL"

        print(f"  {wind_dir:>4} {cx:>+7.2f} {cy:>+7.2f} {angle_measured:>+9.1f}°  "
              f"{angle_expected:>+7.1f}°  {err:>6.1f}°  {ok}")

    print()


if __name__ == "__main__":
    main()
