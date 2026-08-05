"""
Script 09: Métrica de alineación de centroide direccional.

Para cada una de las 8 direcciones de viento (N/NE/E/SE/S/SW/W/NW), toma
muestras reales del val set estratificadas por Dir_viento, genera imágenes
con esas condiciones reales y mide el ángulo del centroide vs. el ángulo
downwind esperado.

Usar condiciones reales (no sintéticas) garantiza que el modelo es evaluado
dentro de la distribución de entrenamiento.

Salidas:
  outputs/metrics/directional_centroid_metric.json
  outputs/figures/directional_centroid_validation.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import get_config
from src.models.generator import Generator

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
_DIR_VIENTO_START = 14

# Ángulo downwind esperado (en grados, convención matemática: E=0°, CCW)
# Viento "desde" D → fuego se propaga en dirección opuesta a D
DOWNWIND_ANGLE = {
    "N":  270.0,   # viento desde el Norte → fuego va al Sur
    "NE": 225.0,
    "E":  180.0,
    "SE": 135.0,
    "S":   90.0,   # viento desde el Sur → fuego va al Norte
    "SW":  45.0,
    "W":    0.0,
    "NW": 315.0,
}


def load_val_conditions_by_direction(cfg) -> dict:
    """Carga el val set y agrupa los índices de registro por Dir_viento.

    Retorna dict {dir_str: list of (condition, mask, veg_profile)}.
    Solo incluye las 8 direcciones físicas (excluye Calma y Missing).
    """
    import json as _json

    condition_vectors  = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)
    masks              = np.load(cfg.paths.masks)

    val_index_path = cfg.paths.ca_images / "val" / "index.json"
    with open(val_index_path) as f:
        val_index = _json.load(f)

    by_dir = {d: [] for d in DIRECTIONS}
    seen_records = set()   # un registro puede tener múltiples imágenes; basta una condición

    for entry in val_index:
        rec_idx = entry["record_idx"]
        if rec_idx in seen_records:
            continue
        seen_records.add(rec_idx)

        onehot = condition_vectors[rec_idx, _DIR_VIENTO_START:_DIR_VIENTO_START + 10]
        dir_idx = int(np.argmax(onehot))
        if dir_idx >= len(DIRECTIONS):   # Calma o Missing
            continue
        wind_dir = DIR_VIENTO_CATS[dir_idx]
        by_dir[wind_dir].append((
            condition_vectors[rec_idx].copy(),
            masks[rec_idx].copy(),
            vegetation_profiles[rec_idx].copy(),
        ))

    for d in DIRECTIONS:
        print(f"  Val {d:3s}: {len(by_dir[d])} registros")

    return by_dir


def compute_centroid_angle(heatmap: np.ndarray, threshold: float = 0.15) -> float | None:
    """
    Calcula el ángulo (grados, conv. matemática E=0 CCW) del centroide de
    píxeles activos respecto al centro del grid.
    Retorna None si no hay suficientes píxeles activos.
    """
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
    dy = -(cy_fire - cy)   # invertir Y: pixel y↓ → matemático y↑
    angle = float(np.degrees(np.arctan2(dy, dx))) % 360
    return angle


def generate_heatmap_from_real_conditions(generator, samples, n_per_sample: int,
                                          device: str, seed: int,
                                          threshold: float = 0.15) -> tuple:
    """Genera imágenes usando condiciones reales del val set.

    Para cada (condition, mask, veg_profile) en `samples`, genera n_per_sample
    realizaciones con z distintos. Calcula el centroide de cada imagen y
    promedia circularmente todos los ángulos.

    Args:
        samples: lista de (condition, mask, veg_profile) — condiciones reales
        n_per_sample: realizaciones por condición (total = len(samples) * n_per_sample)

    Retorna (heatmap_promedio, angulo_medio, std_angulo).
    """
    cfg = get_config()
    torch.manual_seed(seed)

    frames = []
    angles = []

    with torch.no_grad():
        for condition, mask, veg_profile in samples:
            cond_t = torch.tensor(condition,   dtype=torch.float32).unsqueeze(0).to(device)
            mask_t = torch.tensor(mask,        dtype=torch.float32).unsqueeze(0).to(device)
            veg_t  = torch.tensor(veg_profile, dtype=torch.float32).unsqueeze(0).to(device)

            for _ in range(n_per_sample):
                z = torch.randn(1, cfg.model.z_dim, device=device)
                img = generator(z, cond_t, mask_t, veg_t)
                img = (img.squeeze().cpu().numpy() + 1) / 2   # [-1,1] → [0,1]
                img = np.transpose(img, (1, 2, 0))             # CHW → HWC
                r, g = img[:, :, 0], img[:, :, 1]
                fire = np.clip(r - g * 0.7, 0, 1)
                frames.append(fire)
                angle = compute_centroid_angle(fire, threshold=threshold)
                if angle is not None:
                    angles.append(angle)

    mean_heatmap = np.mean(frames, axis=0) if frames else np.zeros((128, 128))

    if not angles:
        return mean_heatmap, None, None

    # Promedio circular de ángulos
    angles_rad = np.radians(angles)
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    mean_angle = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360
    std_angle  = float(np.degrees(np.sqrt(-2 * np.log(
        np.clip(np.sqrt(mean_sin**2 + mean_cos**2), 1e-9, 1.0)
    ))))

    return mean_heatmap, mean_angle, std_angle


def angular_error(measured: float, expected: float) -> float:
    """Diferencia angular mínima en [-180, 180]."""
    diff = (measured - expected + 180) % 360 - 180
    return abs(diff)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=10,
                   help="Registros reales del val set por dirección (default 10). "
                        "Si hay menos, usa todos los disponibles.")
    p.add_argument("--n-per-sample", type=int, default=3,
                   help="Realizaciones z por cada condición real (default 3). "
                        "Total por dirección = n_samples * n_per_sample.")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Umbral de intensidad para definir zona activa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--g-features", type=int, nargs="+", default=None,
                   help="Features del generador (override config). "
                        "Ej: --g-features 512 256 128 64  para 64x64 (Run 20). "
                        "Default: usa cfg.model.g_features.")
    p.add_argument("--run-id", type=str, default=None,
                   help="Identificador del run (ej. run48). Si se indica, el JSON "
                        "se guarda como directional_centroid_metric_<run-id>.json "
                        "además del archivo sin sufijo.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Cargar generador
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    g_features = args.g_features if args.g_features else cfg.model.g_features
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        features=g_features,
        veg_dim=cfg.model.veg_dim,
    ).to(device)
    state = ckpt.get("ema", ckpt.get("generator", ckpt))
    generator.load_state_dict(state)
    generator.eval()
    print(f"Generador cargado (epoch {ckpt.get('epoch', '?')})\n")

    # Cargar condiciones reales del val set estratificadas por dirección
    print("Cargando val set por dirección de viento...")
    by_dir = load_val_conditions_by_direction(cfg)
    print()

    results = {}
    heatmaps = {}

    for wind_dir in DIRECTIONS:
        available = by_dir[wind_dir]
        if not available:
            print(f"  {wind_dir:3s}: sin registros en val set → N/A")
            results[wind_dir] = {
                "measured_angle": None,
                "angle_std_deg": None,
                "expected_angle": DOWNWIND_ANGLE[wind_dir],
                "angular_error_deg": None,
                "n_records": 0,
                "n_realizations": 0,
            }
            heatmaps[wind_dir] = np.zeros((cfg.model.img_size, cfg.model.img_size))
            continue

        # Submuestrear si hay más registros que n_samples
        rng = np.random.default_rng(args.seed)
        if len(available) > args.n_samples:
            idxs = rng.choice(len(available), args.n_samples, replace=False)
            samples = [available[i] for i in idxs]
        else:
            samples = available

        n_real = len(samples)
        n_total = n_real * args.n_per_sample

        hmap, angle, angle_std = generate_heatmap_from_real_conditions(
            generator, samples,
            n_per_sample=args.n_per_sample,
            device=device, seed=args.seed,
            threshold=args.threshold,
        )
        heatmaps[wind_dir] = hmap

        expected = DOWNWIND_ANGLE[wind_dir]
        error = angular_error(angle, expected) if angle is not None else None

        results[wind_dir] = {
            "measured_angle": round(angle, 1) if angle is not None else None,
            "angle_std_deg": round(angle_std, 1) if angle_std is not None else None,
            "expected_angle": expected,
            "angular_error_deg": round(error, 1) if error is not None else None,
            "n_records": n_real,
            "n_realizations": n_total,
        }
        status = f"{error:.1f}°" if error is not None else "NO ACTIVO"
        angle_str = f"{angle:.1f}±{angle_std:.1f}" if angle is not None else "N/A"
        print(f"  {wind_dir:3s}: medido={angle_str}°  esperado={expected:.0f}°  "
              f"error={status}  (n={n_real} registros × {args.n_per_sample} z = {n_total})")

    valid = [r["angular_error_deg"] for r in results.values()
             if r["angular_error_deg"] is not None]
    mean_err = float(np.mean(valid)) if valid else float("nan")
    std_err  = float(np.std(valid))  if valid else float("nan")
    print(f"\nError angular medio: {mean_err:.1f}° ± {std_err:.1f}°")
    print(f"Todas las direcciones correctas (<45°): "
          f"{all(e < 45 for e in valid)}")

    results["summary"] = {
        "mean_angular_error_deg": round(mean_err, 2),
        "std_angular_error_deg":  round(std_err,  2),
        "n_directions": len(valid),
        "n_samples_requested": args.n_samples,
        "n_per_sample": args.n_per_sample,
        "threshold": args.threshold,
        "all_within_45deg": all(e < 45 for e in valid),
        "evaluation_mode": "real_val_conditions",
    }

    # Guardar JSON (siempre al path genérico; además con sufijo de run si se indicó)
    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "directional_centroid_metric.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMétricas guardadas: {json_path}")
    if args.run_id:
        run_json_path = out_dir / f"directional_centroid_metric_{args.run_id}.json"
        with open(run_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Métricas por run guardadas: {run_json_path}")

    # -----------------------------------------------------------------------
    # Figura: polar plot + grid de heatmaps
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))

    # --- Polar plot ---
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    epoch_str = str(ckpt.get('epoch', '?'))
    ax_polar.set_title(
        f"Ángulo centroide vs. downwind esperado\n(epoch {epoch_str}, condiciones reales val set)",
        pad=20, fontsize=11)

    colors = plt.cm.tab10(np.linspace(0, 1, len(DIRECTIONS)))
    for i, wd in enumerate(DIRECTIONS):
        r_data = results[wd]
        expected_rad = np.radians(r_data["expected_angle"])
        measured_rad = np.radians(r_data["measured_angle"]) if r_data["measured_angle"] else None

        ax_polar.annotate("", xy=(expected_rad, 0.85), xytext=(expected_rad, 0),
                          arrowprops=dict(arrowstyle="-|>", color="gray",
                                          lw=1, linestyle="dashed"))
        if measured_rad is not None:
            ax_polar.annotate("", xy=(measured_rad, 0.75), xytext=(measured_rad, 0),
                              arrowprops=dict(arrowstyle="-|>", color=colors[i], lw=2))

    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_rticks([])
    ax_polar.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_polar.set_xticklabels(["E", "NE", "N", "NW", "W", "SW", "S", "SE"], fontsize=9)
    ax_polar.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", lw=1, linestyle="dashed", label="Downwind esperado"),
    ]
    for i, wd in enumerate(DIRECTIONS):
        err = results[wd]["angular_error_deg"]
        if err is not None:
            legend_elements.append(
                Line2D([0], [0], color=colors[i], lw=2,
                       label=f"{wd}  →  {err:.0f}°")
            )
    ax_polar.legend(handles=legend_elements, loc="lower left",
                    bbox_to_anchor=(-0.55, -0.15), fontsize=8.5,
                    title="Dir   Error", title_fontsize=9,
                    framealpha=0.9)

    summary_text = (
        f"Error medio: {mean_err:.1f}° ± {std_err:.1f}°\n"
        f"Todas las direcciones dentro de 45°: {'Sí' if results['summary']['all_within_45deg'] else 'No'}\n"
        f"Modo: condiciones reales val set"
    )
    fig.text(0.25, 0.02, summary_text,
             ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"))

    # --- Título del grid (fuera de los ejes internos) ---
    fig.text(0.73, 0.97, "Heatmaps promedio por dirección de viento\n(condiciones reales val set)",
             ha="center", va="top", fontsize=11)

    # --- Grid de heatmaps ---
    # Espacio disponible: x=[0.52, 0.98], y=[0.08, 0.93]
    # 3 columnas × 3 filas, con hueco en el centro
    cell_w = 0.135
    cell_h = 0.24
    col_starts = [0.525, 0.665, 0.805]
    row_starts  = [0.62, 0.37, 0.10]   # top→bottom: fila 0=NW/N/NE, 1=W/_/E, 2=SW/S/SE

    positions = {
        "NW": (0, 0), "N": (0, 1), "NE": (0, 2),
        "W":  (1, 0),              "E":  (1, 2),
        "SW": (2, 0), "S": (2, 1), "SE": (2, 2),
    }
    for wd, (row, col) in positions.items():
        ax_inner = fig.add_axes([col_starts[col], row_starts[row], cell_w, cell_h])
        ax_inner.imshow(heatmaps[wd], cmap="hot", vmin=0, vmax=0.6, origin="upper")
        err = results[wd]["angular_error_deg"]
        err_str = f"{err:.0f}°" if err is not None else "N/A"
        n_str = f"n={results[wd]['n_records']}"
        ax_inner.set_title(f"{wd}  {err_str}  {n_str}", fontsize=8.5, pad=3)
        ax_inner.set_xticks([]); ax_inner.set_yticks([])
        if results[wd]["measured_angle"] is not None:
            angle_rad = np.radians(results[wd]["measured_angle"])
            H, W_ = heatmaps[wd].shape
            cx_c = W_/2 + np.cos(angle_rad) * W_/4
            cy_c = H/2 - np.sin(angle_rad) * H/4
            ax_inner.plot(cx_c, cy_c, "c+", markersize=8, markeredgewidth=1.5)

    fig.text(0.73, 0.06, "Cruz cyan = centroide medido", ha="center", fontsize=8, color="teal")

    fig_suffix = f"_{args.run_id}" if args.run_id else ""
    fig_path = f"outputs/figures/directional_centroid_validation{fig_suffix}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada: {fig_path}")


if __name__ == "__main__":
    main()
