"""Generación del dataset completo de imágenes del autómata celular."""

import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import Config, get_config
from src.ca.terrain import (
    generate_vegetation_grid, generate_elevation_grid,
    compute_slope, compute_slope_direction,
)
from src.ca.cellular_automaton import run_simulation
from src.ca.renderer import render_to_pil, render_heatmap_to_pil
from src.ca.heatmap import generate_heatmaps_for_record as _generate_heatmaps

# Rangos físicos para normalizar a [0, 1] después de invertir el z-score.
# Basados en el dominio de incendios forestales chilenos (CONAF).
_TEMP_MAX  = 50.0   # °C máximo razonable
_HUM_MAX   = 100.0  # % humedad relativa
_WIND_MAX  = 15.0   # km/h velocidad viento (calibrado al P95 del dataset Maule: 14.3 km/h)


def extract_weather_params(condition_vector: np.ndarray,
                           scaler: StandardScaler = None) -> dict:
    """Extraer parámetros meteorológicos del vector de condición.

    Usa el StandardScaler ajustado en preprocesamiento para invertir el
    z-score y recuperar los valores físicos originales, que luego se
    normalizan a [0, 1] usando rangos de dominio conocidos.

    Si no se provee scaler (retrocompatibilidad), usa el mapeo lineal
    aproximado original sobre el z-score.

    Layout del vector: [4 continuos | 24 one-hot | 4 miss_flags]
    Continuos:
      [0] Temperatura (z-score °C)
      [1] Humedad_Relativa (z-score %)
      [2] Vel_Viento (z-score km/h)
      [3] Sup_Total (z-score de log1p(ha))
    One-hot:
      [4:14]  Exposición (10 cats)
      [14:24] Dir_viento (10 cats)
      [24:28] Topografía (4 cats)
    """
    if scaler is not None:
        # Invertir z-score usando estadísticas reales del scaler
        # scaler fue ajustado sobre [Temperatura, Humedad, Vel_Viento, log1p(Sup_Total)]
        z = condition_vector[:4].astype(np.float64)
        physical = z * scaler.scale_[:4] + scaler.mean_[:4]
        # physical[3] es log1p(area), no lo usamos para el CA directamente
        temp_raw  = float(physical[0])
        hum_raw   = float(physical[1])
        wind_raw  = float(physical[2])
        temperature = float(np.clip(temp_raw  / _TEMP_MAX,  0.0, 1.0))
        humidity    = float(np.clip(hum_raw   / _HUM_MAX,   0.0, 1.0))
        wind_speed  = float(np.clip(wind_raw  / _WIND_MAX,  0.0, 1.0))
    else:
        # Fallback: mapeo lineal aproximado sobre z-scores (comportamiento anterior)
        temperature = float(np.clip((condition_vector[0] + 2) / 4, 0, 1))
        humidity    = float(np.clip((condition_vector[1] + 2) / 4, 0, 1))
        wind_speed  = float(np.clip((condition_vector[2] + 1) / 4, 0, 1))

    # Dirección del viento: decodificar one-hot
    wind_dir_cats = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
    wind_onehot = condition_vector[14:24]
    wind_dir = wind_dir_cats[int(np.argmax(wind_onehot))]

    # Topografía: decodificar one-hot
    topo_cats = ["Suave", "Irregular", "Abrupta", "Missing"]
    topo_onehot = condition_vector[24:28]
    topography = topo_cats[int(np.argmax(topo_onehot))]

    # Superficie total: z-score de log1p(ha) en índice 3
    sup_total_z = float(condition_vector[3])

    return {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "topography": topography,
        "sup_total_z": sup_total_z,
    }


def generate_images_for_record(
    record_idx: int,
    condition_vector: np.ndarray,
    veg_profile: np.ndarray,
    cfg: Config,
    output_dir: Path,
    base_seed: int = 0,
    scaler: StandardScaler = None,
) -> list:
    """Generar imágenes del CA para un registro.

    Si heatmap_mode está activo, genera heatmaps Monte Carlo.
    Si no, genera snapshots discretos del CA.

    Args:
        scaler: StandardScaler ajustado en preprocesamiento. Cuando se provee,
                los z-scores se invierten correctamente a valores físicos antes
                de pasarlos al CA. Si es None, usa mapeo aproximado (legado).

    Returns:
        Lista de dicts con metadata de cada imagen generada.
    """
    ca_cfg = cfg.ca
    params = extract_weather_params(condition_vector, scaler=scaler)

    if ca_cfg.heatmap_mode:
        return _generate_heatmap_images(
            record_idx, veg_profile, params, cfg, output_dir, base_seed
        )
    else:
        return _generate_snapshot_images(
            record_idx, veg_profile, params, cfg, output_dir, base_seed
        )


def _generate_heatmap_images(
    record_idx: int, veg_profile: np.ndarray, params: dict,
    cfg: Config, output_dir: Path, base_seed: int,
    **_kwargs,
) -> list:
    """Generar heatmaps Monte Carlo para un registro."""
    ca_cfg = cfg.ca
    entries = []

    results = _generate_heatmaps(
        veg_profile=veg_profile,
        wind_dir=params["wind_dir"],
        wind_speed=params["wind_speed"],
        humidity=params["humidity"],
        temperature=params["temperature"],
        topography=params["topography"],
        cfg=ca_cfg,
        n_runs=ca_cfg.n_monte_carlo_runs,
        n_horizons=ca_cfg.n_horizons,
        n_terrain_variations=ca_cfg.n_terrain_variations,
        base_seed=base_seed + record_idx * 100,
        sup_total_z=params["sup_total_z"],
    )

    min_coverage = cfg.ca.min_fire_coverage
    for heatmap, veg_grid, var_idx, h_idx in results:
        if heatmap.mean() < min_coverage:
            continue  # descartar imágenes con fuego insuficiente
        img = render_heatmap_to_pil(heatmap, veg_grid)
        filename = f"rec{record_idx:05d}_var{var_idx}_h{h_idx}.png"
        img.save(output_dir / filename)

        entries.append({
            "filename": filename,
            "record_idx": record_idx,
            "variation": var_idx,
            "horizon": h_idx,
        })

    return entries


def _generate_snapshot_images(
    record_idx: int, veg_profile: np.ndarray, params: dict,
    cfg: Config, output_dir: Path, base_seed: int,
) -> list:
    """Generar snapshots discretos del CA para un registro (modo original)."""
    ca_cfg = cfg.ca
    entries = []

    for var_idx in range(ca_cfg.n_terrain_variations):
        seed = base_seed + record_idx * 100 + var_idx

        veg_grid = generate_vegetation_grid(veg_profile, ca_cfg, seed=seed)
        elevation = generate_elevation_grid(params["topography"], ca_cfg, seed=seed)
        slope = compute_slope(elevation)
        slope_dir = compute_slope_direction(elevation)

        snapshots, _ = run_simulation(
            veg_grid=veg_grid, elevation=elevation,
            slope=slope, slope_dir=slope_dir,
            wind_dir=params["wind_dir"],
            wind_speed=params["wind_speed"],
            humidity=params["humidity"],
            temperature=params["temperature"],
            cfg=ca_cfg, seed=seed,
        )

        for snap_idx, state in enumerate(snapshots):
            step = int((snap_idx + 1) / len(snapshots) * ca_cfg.n_steps)
            img = render_to_pil(state, veg_grid, step, ca_cfg.n_steps)
            filename = f"rec{record_idx:05d}_var{var_idx}_step{snap_idx}.png"
            img.save(output_dir / filename)

            entries.append({
                "filename": filename,
                "record_idx": record_idx,
                "variation": var_idx,
                "snapshot": snap_idx,
                "step": step,
            })

    return entries


def _worker_generate(args):
    """Worker para multiprocessing. Genera imágenes para un registro."""
    rec_idx, condition_vector, veg_profile, cfg, output_dir, scaler = args
    return generate_images_for_record(
        record_idx=rec_idx,
        condition_vector=condition_vector,
        veg_profile=veg_profile,
        cfg=cfg,
        output_dir=output_dir,
        scaler=scaler,
    )


def generate_full_dataset(cfg: Config = None, max_records: int = None,
                          n_workers: int = None):
    """Generar dataset completo de imágenes del CA.

    Args:
        cfg: configuración
        max_records: límite de registros (para pruebas rápidas)
        n_workers: número de procesos paralelos (default: CPU cores - 1)
    """
    if cfg is None:
        cfg = get_config()

    # Cargar datos preprocesados
    condition_vectors = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)

    # Cargar scaler para invertir z-scores correctamente en el CA
    scaler = None
    if cfg.paths.scaler_path.exists():
        with open(cfg.paths.scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"  Scaler cargado desde {cfg.paths.scaler_path}")
    else:
        print(f"  AVISO: scaler no encontrado en {cfg.paths.scaler_path}. "
              "Usando mapeo aproximado de z-scores.")

    n_records = len(condition_vectors)
    if max_records is not None:
        n_records = min(n_records, max_records)

    # Split train/val
    rng = np.random.default_rng(cfg.data.seed)
    indices = np.arange(n_records)
    rng.shuffle(indices)
    n_val = int(n_records * cfg.data.val_fraction)
    val_indices = set(indices[:n_val])

    train_dir = cfg.paths.ca_images / "train"
    val_dir = cfg.paths.ca_images / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    if cfg.ca.heatmap_mode:
        imgs_per_record = cfg.ca.n_terrain_variations * cfg.ca.n_horizons
        mode_desc = (f"{cfg.ca.n_terrain_variations} variaciones × "
                     f"{cfg.ca.n_horizons} horizontes × "
                     f"{cfg.ca.n_monte_carlo_runs} MC runs")
    else:
        imgs_per_record = cfg.ca.n_terrain_variations * cfg.ca.n_snapshots
        mode_desc = (f"{cfg.ca.n_terrain_variations} variaciones × "
                     f"{cfg.ca.n_snapshots} snapshots")

    total_imgs = n_records * imgs_per_record

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Modo: {'Heatmap Monte Carlo' if cfg.ca.heatmap_mode else 'Snapshots discretos'}")
    print(f"Generando {total_imgs} imágenes para {n_records} registros...")
    print(f"  {n_records - len(val_indices)} train + {len(val_indices)} val")
    print(f"  {mode_desc}")
    print(f"  Workers: {n_workers}")

    # Preparar argumentos para cada registro
    work_items = []
    for rec_idx in range(n_records):
        output_dir = val_dir if rec_idx in val_indices else train_dir
        work_items.append((
            rec_idx,
            condition_vectors[rec_idx],
            vegetation_profiles[rec_idx],
            cfg,
            output_dir,
            scaler,
        ))

    train_entries = []
    val_entries = []

    if n_workers <= 1:
        # Modo secuencial
        for args in tqdm(work_items, desc="Generando imágenes CA"):
            entries = _worker_generate(args)
            rec_idx = args[0]
            if rec_idx in val_indices:
                val_entries.extend(entries)
            else:
                train_entries.extend(entries)
    else:
        # Modo paralelo
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(_worker_generate, work_items),
                total=len(work_items),
                desc=f"Generando imágenes CA ({n_workers} workers)",
            ))

        for args, entries in zip(work_items, results):
            rec_idx = args[0]
            if rec_idx in val_indices:
                val_entries.extend(entries)
            else:
                train_entries.extend(entries)

    # Guardar índices
    with open(train_dir / "index.json", "w") as f:
        json.dump(train_entries, f)
    with open(val_dir / "index.json", "w") as f:
        json.dump(val_entries, f)

    print(f"Dataset generado: {len(train_entries)} train, {len(val_entries)} val")
    return train_entries, val_entries
