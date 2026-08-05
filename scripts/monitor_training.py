#!/usr/bin/env python3
"""
Monitor en tiempo real del entrenamiento del GAN.

Analiza los grids de samples generados en outputs/samples/ para detectar:
  - Colapso de modos (mode collapse)
  - Falta de diversidad entre imágenes
  - Imágenes planas/uniformes (sin textura)

Uso:
  python scripts/monitor_training.py
  python scripts/monitor_training.py --interval 30 --samples-dir outputs/samples
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# ── Colores ANSI ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Parámetros del grid (deben coincidir con SamplingCallback) ─────────────────
IMG_SIZE = 64   # tamaño de cada imagen generada
PADDING  = 2    # padding entre imágenes (make_grid padding=2)
N_ROW    = 8    # columnas del grid


def extract_patches(grid_img: np.ndarray) -> np.ndarray:
    """Extrae imágenes individuales del grid generado por torchvision.make_grid."""
    H, W, C = grid_img.shape
    cell = IMG_SIZE + PADDING          # tamaño de celda = img + padding derecho/inferior
    n_cols = (W - PADDING) // cell     # = nrow = 8
    n_rows = (H - PADDING) // cell

    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            y = PADDING + r * cell
            x = PADDING + c * cell
            if y + IMG_SIZE <= H and x + IMG_SIZE <= W:
                patches.append(grid_img[y:y + IMG_SIZE, x:x + IMG_SIZE])

    return np.array(patches, dtype=np.float32)   # (N, H, W, C)


def analyze_grid(path: Path) -> dict:
    """Carga un grid PNG y calcula métricas de calidad/diversidad."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    patches = extract_patches(img)
    n = len(patches)

    if n == 0:
        return {"error": "No se pudieron extraer patches del grid"}

    flat = patches.reshape(n, -1)   # (N, H*W*C)

    # ── Diversidad inter-muestras ─────────────────────────────────────────────
    # Std por posición de píxel entre todas las muestras, luego promedio.
    # Valor bajo → todas las imágenes son iguales (mode collapse).
    diversity = float(patches.std(axis=0).mean())

    # ── Textura intra-muestra ─────────────────────────────────────────────────
    # Std de píxeles dentro de cada imagen individual, luego promedio.
    # Valor bajo → cada imagen es un color/tono uniforme.
    texture = float(flat.std(axis=1).mean())

    # ── Rango dinámico ────────────────────────────────────────────────────────
    pmin = float(patches.min())
    pmax = float(patches.max())
    drange = pmax - pmin

    # ── Similitud media entre pares (cosine) ──────────────────────────────────
    # Valor alto (> 0.95) indica que las muestras son casi idénticas.
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    normed = flat / np.clip(norms, 1e-8, None)
    # Usar subconjunto de 16 para eficiencia
    sub = normed[:min(n, 16)]
    sim_matrix = sub @ sub.T
    mask = ~np.eye(len(sub), dtype=bool)
    mean_sim = float(sim_matrix[mask].mean())

    # ── Entropía del histograma (diversidad global) ───────────────────────────
    hist, _ = np.histogram(patches, bins=64, range=(0.0, 1.0))
    hist = hist / hist.sum()
    entropy = float(-np.sum(hist * np.log(hist + 1e-10)))

    return {
        "n_samples": n,
        "diversity": diversity,
        "texture": texture,
        "min_pixel": pmin,
        "max_pixel": pmax,
        "dynamic_range": drange,
        "mean_sim": mean_sim,
        "entropy": entropy,
    }


def classify_health(m: dict):
    """Devuelve (status, lista_de_problemas) según las métricas."""
    issues = []

    # Colapso de modos – métrica principal
    if m["diversity"] < 0.015:
        issues.append(("CRITICO", f"Mode collapse: diversidad={m['diversity']:.4f} (casi cero)"))
    elif m["diversity"] < 0.04:
        issues.append(("AVISO", f"Baja diversidad: {m['diversity']:.4f}"))

    # Imágenes planas
    if m["texture"] < 0.02:
        issues.append(("CRITICO", f"Imágenes sin textura (planas): texture={m['texture']:.4f}"))
    elif m["texture"] < 0.05:
        issues.append(("AVISO", f"Textura baja: {m['texture']:.4f}"))

    # Rango dinámico colapsado
    if m["dynamic_range"] < 0.10:
        issues.append(("CRITICO", f"Rango dinámico colapsado: [{m['min_pixel']:.2f}, {m['max_pixel']:.2f}]"))
    elif m["dynamic_range"] < 0.30:
        issues.append(("AVISO", f"Rango dinámico reducido: {m['dynamic_range']:.2f}"))

    # Similitud entre pares
    if m["mean_sim"] > 0.97:
        issues.append(("CRITICO", f"Muestras casi idénticas: sim={m['mean_sim']:.4f}"))
    elif m["mean_sim"] > 0.90:
        issues.append(("AVISO", f"Alta similitud entre muestras: {m['mean_sim']:.4f}"))

    if not issues:
        return "OK", []
    severity = "CRITICO" if any(s == "CRITICO" for s, _ in issues) else "AVISO"
    return severity, issues


def read_last_csv_row(csv_path: Path) -> dict | None:
    """Lee la última fila del CSV de métricas de entrenamiento."""
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def epoch_from_filename(p: Path) -> int:
    """Extrae el número de época del nombre 'samples_epochXXXX.png'."""
    try:
        return int(p.stem.replace("samples_epoch", ""))
    except ValueError:
        return -1


def bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    """Barra ASCII normalizada entre lo y hi."""
    ratio = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(ratio * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def print_report(epoch: int, m: dict, status: str, issues: list,
                 csv_row: dict | None):
    ts = datetime.now().strftime("%H:%M:%S")

    # Color de estado
    c_status = GREEN if status == "OK" else (YELLOW if status == "AVISO" else RED)
    label = f"{BOLD}{c_status}[{status}]{RESET}"

    print(f"\n{'─'*70}")
    print(f"  {ts}  Época {BOLD}{epoch:>4d}{RESET}  {label}")
    print(f"{'─'*70}")

    # Métricas de imagen
    d = m["diversity"]
    t = m["texture"]
    s = m["mean_sim"]
    e = m["entropy"]

    c_d = GREEN if d > 0.06 else (YELLOW if d > 0.03 else RED)
    c_t = GREEN if t > 0.08 else (YELLOW if t > 0.04 else RED)
    c_s = GREEN if s < 0.85 else (YELLOW if s < 0.92 else RED)

    print(f"  {CYAN}Imagen{RESET}  ({m['n_samples']} muestras)")
    print(f"    Diversidad  {c_d}{d:.4f}{RESET}  {bar(d, 0, 0.15)}  (> 0.06 OK)")
    print(f"    Textura     {c_t}{t:.4f}{RESET}  {bar(t, 0, 0.20)}  (> 0.08 OK)")
    print(f"    Similitud   {c_s}{s:.4f}{RESET}  {bar(1-s, 0, 1)}  (< 0.85 OK)")
    print(f"    Rango px    [{m['min_pixel']:.3f}, {m['max_pixel']:.3f}]   "
          f"Δ={m['dynamic_range']:.3f}   Entropía={e:.2f}")

    # Métricas de entrenamiento (CSV)
    if csv_row:
        print(f"\n  {CYAN}Entrenamiento{RESET}  (época {csv_row['epoch']})")
        dl = float(csv_row["d_loss"])
        gl = float(csv_row["g_loss"])
        dr = float(csv_row["d_real"])
        df = float(csv_row["d_fake"])
        r1 = float(csv_row["r1"])

        c_dl = GREEN if 0.8 < dl < 2.0 else (YELLOW if dl < 3.0 else RED)
        c_gl = GREEN if 0.3 < gl < 1.5 else YELLOW
        c_dr = GREEN if 0.0 < dr < 1.0 else YELLOW
        c_df = GREEN if -1.0 < df < 0.0 else YELLOW

        print(f"    D_loss={c_dl}{dl:.4f}{RESET}   G_loss={c_gl}{gl:.4f}{RESET}")
        print(f"    D(real)={c_dr}{dr:+.4f}{RESET}  D(fake)={c_df}{df:+.4f}{RESET}   R1={r1:.5f}")

    # Problemas detectados
    if issues:
        print()
        for sev, msg in issues:
            c = RED if sev == "CRITICO" else YELLOW
            print(f"  {c}{BOLD}>> {sev}:{RESET} {msg}")

    print(f"{'─'*70}")


def watch(samples_dir: Path, csv_path: Path, interval: int, start_time: float):
    """Bucle principal: detecta nuevos archivos y los analiza."""
    # Usar str resuelto para evitar comparaciones erróneas en Windows
    seen: set[str] = set()

    # Ignorar archivos anteriores al inicio del script
    for p in samples_dir.glob("samples_epoch*.png"):
        if p.stat().st_mtime < start_time:
            seen.add(str(p.resolve()))

    print(f"{BOLD}Monitor activo{RESET} — esperando nuevos samples en {samples_dir}/")
    print(f"{DIM}Intervalo de sondeo: {interval}s   Ctrl+C para salir{RESET}")

    epochs_seen = []
    metrics_history = []

    while True:
        new_files = sorted(
            [p for p in samples_dir.glob("samples_epoch*.png")
             if str(p.resolve()) not in seen],
            key=epoch_from_filename,
        )

        for path in new_files:
            seen.add(str(path.resolve()))
            epoch = epoch_from_filename(path)
            if epoch < 0:
                continue

            m = analyze_grid(path)
            if "error" in m:
                print(f"{RED}Error analizando {path.name}: {m['error']}{RESET}")
                continue

            status, issues = classify_health(m)
            csv_row = read_last_csv_row(csv_path)

            print_report(epoch, m, status, issues, csv_row)

            # Tendencia de diversidad
            epochs_seen.append(epoch)
            metrics_history.append(m["diversity"])
            if len(metrics_history) >= 3:
                trend = np.polyfit(epochs_seen[-5:], metrics_history[-5:], 1)[0]
                arrow = "↑" if trend > 0.0002 else ("↓" if trend < -0.0002 else "→")
                c_arrow = GREEN if trend > 0 else (RED if trend < -0.0002 else YELLOW)
                print(f"  Tendencia diversidad: {c_arrow}{arrow} {trend:+.5f}/época{RESET}")

        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description="Monitor de entrenamiento GAN (análisis de samples)")
    ap.add_argument("--samples-dir", default="outputs/samples",
                    help="Directorio con los grids PNG generados")
    ap.add_argument("--csv", default="outputs/metrics/training_log.csv",
                    help="CSV de métricas de entrenamiento")
    ap.add_argument("--interval", type=int, default=20,
                    help="Segundos entre sondeos del directorio (default: 20)")
    ap.add_argument("--retroactive", action="store_true",
                    help="Analizar también los samples existentes antes de arrancar")
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)
    csv_path    = Path(args.csv)

    if not samples_dir.exists():
        samples_dir.mkdir(parents=True)
        print(f"{YELLOW}Directorio creado: {samples_dir}{RESET}")

    start_time = 0.0 if args.retroactive else time.time()
    try:
        watch(samples_dir, csv_path, args.interval, start_time=start_time)
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor detenido.{RESET}")


if __name__ == "__main__":
    main()
