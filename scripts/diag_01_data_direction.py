#!/usr/bin/env python3
"""Diagnóstico 1: ¿Los datos CA tienen señal direccional?

Para cada dirección de viento (0=N..7=NW), carga imágenes del val set,
calcula el centroide del fuego y mide el error angular respecto al
downwind esperado. Si los datos no tienen señal, el modelo no puede aprender.
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config

# Vectores downwind esperados (imagen coords: y hacia abajo)
# 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
_SQRT2 = math.sqrt(2) / 2.0
_DIR_NAMES  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_DIR_TARGET = [
    ( 0.0,   +1.0  ),  # N   → fuego va hacia abajo (y+)
    (-_SQRT2,+_SQRT2), # NE
    (-1.0,    0.0  ),  # E   → fuego va hacia izquierda (x-)
    (-_SQRT2,-_SQRT2), # SE
    ( 0.0,   -1.0  ),  # S   → fuego va hacia arriba (y-)
    (+_SQRT2,-_SQRT2), # SW
    (+1.0,    0.0  ),  # W   → fuego va hacia derecha (x+)
    (+_SQRT2,+_SQRT2), # NW
]

_DIR_VIENTO_START = 14


def load_val_images_by_dir(cfg, max_per_dir=100):
    """Carga imágenes val agrupadas por dirección de viento."""
    import json

    condition_vectors = np.load(cfg.paths.condition_vectors)
    val_dir = cfg.paths.ca_images / "val"
    index_file = val_dir / "index.json"

    transform = T.Compose([
        T.Resize((cfg.model.img_size, cfg.model.img_size)),
        T.ToTensor(),
    ])

    with open(index_file) as f:
        index = json.load(f)

    by_dir = defaultdict(list)
    for entry in index:
        rec_idx = entry["record_idx"]
        onehot = condition_vectors[rec_idx, _DIR_VIENTO_START:_DIR_VIENTO_START + 8]
        if onehot.sum() < 0.5:
            continue  # Calma o Missing
        d = int(np.argmax(onehot))
        if len(by_dir[d]) >= max_per_dir:
            continue
        img_path = val_dir / entry["filename"]
        img = Image.open(img_path).convert("RGB")
        img_t = transform(img)  # [3, H, W] en [0,1]
        by_dir[d].append(img_t)

    return by_dir


def compute_mean_centroid(imgs):
    """Dado un stack de imágenes [N,3,H,W] en [0,1], calcula centroide promedio (cx, cy).

    Promedia los vectores centroide, NO los ángulos individuales.
    Los ángulos individuales son ruido cuando el fuego está cerca del centro (cx≈0).
    """
    imgs_t = torch.stack(imgs)  # [N, 3, H, W]
    fire = imgs_t.mean(dim=1)   # [N, H, W] — intensidad
    B, H, W = fire.shape
    yy = torch.arange(H, dtype=torch.float32) - (H - 1) / 2.0
    xx = torch.arange(W, dtype=torch.float32) - (W - 1) / 2.0
    mass = fire.sum(dim=(1, 2)).clamp(min=1e-6)
    cy = (fire * yy.view(1, H, 1)).sum(dim=(1, 2)) / mass  # [N]
    cx = (fire * xx.view(1, 1, W)).sum(dim=(1, 2)) / mass  # [N]
    # Promedio de vectores (no de ángulos)
    mean_cx = cx.mean().item()
    mean_cy = cy.mean().item()
    std_cx = cx.std().item()
    std_cy = cy.std().item()
    angle = math.degrees(math.atan2(mean_cy, mean_cx))
    return mean_cx, mean_cy, std_cx, std_cy, angle


def angular_error(measured_deg, expected_tx, expected_ty):
    """Error angular mínimo entre ángulo medido y vector target."""
    expected_deg = math.degrees(math.atan2(expected_ty, expected_tx))
    diff = measured_deg - expected_deg
    # Normalizar a [-180, 180]
    diff = (diff + 180) % 360 - 180
    return abs(diff)


def main():
    cfg = get_config()
    print("Cargando val set...")
    by_dir = load_val_images_by_dir(cfg, max_per_dir=200)

    print(f"\n{'Dir':>4} {'N imgs':>7} {'cx_mean':>9} {'cy_mean':>9} {'Angulo':>8} {'Esperado':>10} {'Error':>8} {'OK?':>5}")
    print("-" * 70)

    errors = []
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for d in range(8):
        imgs = by_dir.get(d, [])
        if not imgs:
            print(f"  {_DIR_NAMES[d]:>4} {'0':>7}")
            continue

        mean_cx, mean_cy, std_cx, std_cy, mean_angle = compute_mean_centroid(imgs)

        tx, ty = _DIR_TARGET[d]
        expected_deg = math.degrees(math.atan2(ty, tx))
        err = angular_error(mean_angle, tx, ty)
        ok  = "OK" if err < 45 else "FAIL"
        errors.append(err)

        print(f"  {_DIR_NAMES[d]:>4} {len(imgs):>7} {mean_cx:>+7.2f}±{std_cx:4.2f} "
              f"{mean_cy:>+7.2f}±{std_cy:4.2f} "
              f"{mean_angle:>+7.1f}°  {expected_deg:>+7.1f}°  {err:>6.1f}°  {ok}")

        # Promedio visual de imágenes
        ax = axes[d]
        mean_img = torch.stack(imgs).mean(dim=0).permute(1, 2, 0).numpy()
        ax.imshow(mean_img, cmap="hot")
        ax.set_title(f"{_DIR_NAMES[d]}: err={err:.1f}°  n={len(imgs)}", fontsize=9)
        ax.axis("off")

    if errors:
        print(f"\n  Error medio: {np.mean(errors):.1f}° ± {np.std(errors):.1f}°")
        n_ok = sum(1 for e in errors if e < 45)
        print(f"  Direcciones correctas (<45°): {n_ok}/{len(errors)}")

    out_path = cfg.paths.figures / "diag01_data_direction.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    print(f"\n  Figura guardada: {out_path}")


if __name__ == "__main__":
    main()
