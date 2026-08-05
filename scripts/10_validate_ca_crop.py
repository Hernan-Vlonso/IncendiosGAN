"""
Script 10: Valida señal direccional del CA con grid 2× y recorte central.

Corre el CA en grid 128×128 (doble del output deseado 64×64),
recorta el centro 64×64 y mide el centroide de propagación por cada
dirección de viento. Si el CA recortado tiene señal limpia, el problema
de Run 25 era truncamiento en el borde y Run 26 (dataset con crop) está
justificado.

Uso:
    python scripts/10_validate_ca_crop.py
    python scripts/10_validate_ca_crop.py --n-records 5 --n-mc-runs 15
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import get_config
from src.ca.heatmap import run_monte_carlo
from src.ca.terrain import (
    generate_vegetation_grid, generate_elevation_grid,
    compute_slope, compute_slope_direction,
)
from src.ca.generate_dataset import extract_weather_params
import pickle

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
_DIR_VIENTO_START = 14

DOWNWIND_ANGLE = {
    "N":  270.0,
    "NE": 225.0,
    "E":  180.0,
    "SE": 135.0,
    "S":   90.0,
    "SW":  45.0,
    "W":    0.0,
    "NW": 315.0,
}


def compute_centroid_angle(heatmap: np.ndarray, threshold: float = 0.15):
    H, W = heatmap.shape
    cx, cy = W / 2.0, H / 2.0
    active = heatmap > threshold
    if active.sum() < 10:
        return None
    ys, xs = np.where(active)
    weights = heatmap[active]
    cx_fire = float(np.average(xs, weights=weights))
    cy_fire = float(np.average(ys, weights=weights))
    dx = cx_fire - cx
    dy = -(cy_fire - cy)
    return float(np.degrees(np.arctan2(dy, dx))) % 360


def angular_error(measured: float, expected: float) -> float:
    diff = (measured - expected + 180) % 360 - 180
    return abs(diff)


def load_val_records_by_direction(cfg, condition_vectors) -> dict:
    """Carga record_idx del val set agrupados por Dir_viento."""
    val_index_path = cfg.paths.ca_images / "val" / "index.json"
    with open(val_index_path) as f:
        val_index = json.load(f)

    by_dir = {d: [] for d in DIRECTIONS}
    seen = set()
    for entry in val_index:
        rec_idx = entry["record_idx"]
        if rec_idx in seen:
            continue
        seen.add(rec_idx)
        onehot = condition_vectors[rec_idx, _DIR_VIENTO_START:_DIR_VIENTO_START + 10]
        dir_idx = int(np.argmax(onehot))
        if dir_idx < len(DIRECTIONS):
            by_dir[DIR_VIENTO_CATS[dir_idx]].append(rec_idx)
    return by_dir


def run_ca_and_crop(rec_idx, condition_vectors, vegetation_profiles, scaler, cfg_ca,
                    grid_large: int, crop_size: int,
                    n_mc_runs: int, seed: int) -> np.ndarray:
    """Corre CA en grid_large×grid_large y recorta centro crop_size×crop_size.

    Retorna heatmap del último horizonte recortado.
    """
    import copy

    # Config CA con grid grande
    cfg_large = copy.copy(cfg_ca)
    cfg_large.grid_size = grid_large

    params = extract_weather_params(condition_vectors[rec_idx], scaler)
    veg_profile = vegetation_profiles[rec_idx]

    veg_grid  = generate_vegetation_grid(veg_profile, cfg_large, seed=seed)
    elevation = generate_elevation_grid(params["topography"], cfg_large, seed=seed)
    slope     = compute_slope(elevation)
    slope_dir = compute_slope_direction(elevation)


    heatmaps = run_monte_carlo(
        veg_grid=veg_grid,
        elevation=elevation,
        slope=slope,
        slope_dir=slope_dir,
        wind_dir=params["wind_dir"],
        wind_speed=params["wind_speed"],
        humidity=params["humidity"],
        temperature=params["temperature"],
        cfg=cfg_large,
        n_runs=n_mc_runs,
        base_seed=seed,
        n_horizons=cfg_large.n_horizons,
        sup_total_z=float(condition_vectors[rec_idx][3]),
    )

    # Recortar centro crop_size×crop_size del último horizonte (mayor propagación)
    hmap = heatmaps[-1]  # shape (grid_large, grid_large)
    margin = (grid_large - crop_size) // 2
    cropped = hmap[margin:margin + crop_size, margin:margin + crop_size]
    return cropped


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-records", type=int, default=8,
                   help="Registros por dirección (default 8)")
    p.add_argument("--n-mc-runs", type=int, default=20,
                   help="Corridas Monte Carlo por registro (default 20)")
    p.add_argument("--grid-large", type=int, default=128,
                   help="Tamaño del grid CA grande (default 128 = 2× de 64)")
    p.add_argument("--crop-size", type=int, default=64,
                   help="Tamaño del recorte central (default 64)")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    print(f"Validacion CA con grid {args.grid_large}x{args.grid_large} -> recorte {args.crop_size}x{args.crop_size}")
    print(f"Registros por dirección: {args.n_records} | MC runs: {args.n_mc_runs}\n")

    condition_vectors   = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)
    with open(cfg.paths.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    by_dir = load_val_records_by_direction(cfg, condition_vectors)

    results = {}
    heatmaps_plot = {}
    rng = np.random.default_rng(args.seed)

    for wind_dir in DIRECTIONS:
        available = by_dir[wind_dir]
        if not available:
            print(f"  {wind_dir:3s}: sin registros en val set")
            results[wind_dir] = None
            heatmaps_plot[wind_dir] = np.zeros((args.crop_size, args.crop_size))
            continue

        n = min(args.n_records, len(available))
        idxs = rng.choice(len(available), n, replace=False)
        records = [available[i] for i in idxs]

        crops = []
        angles = []
        for i, rec_idx in enumerate(records):
            seed_i = args.seed + i
            crop = run_ca_and_crop(
                rec_idx, condition_vectors, vegetation_profiles, scaler, cfg.ca,
                grid_large=args.grid_large,
                crop_size=args.crop_size,
                n_mc_runs=args.n_mc_runs,
                seed=seed_i,
            )
            crops.append(crop)
            angle = compute_centroid_angle(crop, threshold=args.threshold)
            if angle is not None:
                angles.append(angle)

        mean_heatmap = np.mean(crops, axis=0)
        heatmaps_plot[wind_dir] = mean_heatmap

        if not angles:
            print(f"  {wind_dir:3s}: sin píxeles activos en recorte")
            results[wind_dir] = {"measured": None, "expected": DOWNWIND_ANGLE[wind_dir], "error": None}
            continue

        angles_rad = np.radians(angles)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        mean_angle = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360
        std_angle  = float(np.degrees(np.sqrt(-2 * np.log(
            np.clip(np.sqrt(mean_sin**2 + mean_cos**2), 1e-9, 1.0)
        ))))

        expected = DOWNWIND_ANGLE[wind_dir]
        error = angular_error(mean_angle, expected)
        ok = "✓" if error < 45 else "✗"
        print(f"  {wind_dir:3s}: medido={mean_angle:.1f}±{std_angle:.1f}°  "
              f"esperado={expected:.0f}°  error={error:.1f}°  {ok}  (n={n})")
        results[wind_dir] = {"measured": round(mean_angle, 1), "expected": expected,
                              "error": round(error, 1), "std": round(std_angle, 1)}

    # Resumen
    valid_errors = [r["error"] for r in results.values() if r and r["error"] is not None]
    passed = sum(1 for e in valid_errors if e < 45)
    print(f"\nDirecciones correctas (<45°): {passed}/{len(valid_errors)}")
    print(f"Error medio: {np.mean(valid_errors):.1f}° ± {np.std(valid_errors):.1f}°")

    # Comparar con Run 25 (sin crop) — valores de referencia
    run25_errors = {"N": 46.8, "NE": 28.3, "E": 10.4, "SE": 25.3,
                    "S": 40.2, "SW": 82.7, "W": 77.4, "NW": 70.7}
    print(f"\nComparación con Run 25 (sin crop, CA 64×64 fijo):")
    print(f"  {'Dir':4s}  {'CA recortado':>14s}  {'Run 25':>8s}  {'Mejora':>8s}")
    for d in DIRECTIONS:
        r = results.get(d)
        err_new = r["error"] if r and r["error"] is not None else None
        err_old = run25_errors.get(d)
        if err_new is not None and err_old is not None:
            delta = err_old - err_new
            sign = "+" if delta > 0 else ""
            print(f"  {d:4s}  {err_new:>10.1f}°     {err_old:>6.1f}°   {sign}{delta:.1f}°")
        else:
            print(f"  {d:4s}  {'N/A':>14s}  {err_old:>8.1f}°")

    # Figura
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"CA heatmaps recortados ({args.grid_large}×{args.grid_large} → {args.crop_size}×{args.crop_size})\n"
                 f"n={args.n_records} registros × {args.n_mc_runs} MC runs por dirección", fontsize=11)
    for ax, wind_dir in zip(axes.flat, DIRECTIONS):
        hmap = heatmaps_plot[wind_dir]
        ax.imshow(hmap, cmap="hot", vmin=0, vmax=0.6, origin="upper")
        r = results.get(wind_dir)
        if r and r.get("error") is not None:
            err_str = f"{r['error']:.0f}°"
            ok = "✓" if r["error"] < 45 else "✗"
            ax.set_title(f"{wind_dir}  {err_str} {ok}", fontsize=10)
            # Marcar centroide
            angle_rad = np.radians(r["measured"])
            H, W = hmap.shape
            cx_c = W/2 + np.cos(angle_rad) * W/4
            cy_c = H/2 - np.sin(angle_rad) * H/4
            ax.plot(cx_c, cy_c, "c+", markersize=10, markeredgewidth=2)
            # Marcar esperado
            exp_rad = np.radians(r["expected"])
            cx_e = W/2 + np.cos(exp_rad) * W/4
            cy_e = H/2 - np.sin(exp_rad) * H/4
            ax.plot(cx_e, cy_e, "g+", markersize=10, markeredgewidth=2)
        else:
            ax.set_title(f"{wind_dir}  N/A", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.text(0.5, 0.01, "Cyan = centroide medido  |  Verde = downwind esperado",
             ha="center", fontsize=9)
    out_path = "outputs/figures/ca_crop_validation.png"
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nFigura guardada: {out_path}")


if __name__ == "__main__":
    main()
