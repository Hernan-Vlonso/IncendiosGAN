#!/usr/bin/env python3
"""Diagnóstico 4: Regenerar muestra de imágenes E y comparar con las guardadas.

Si las imágenes guardadas tienen bug pero el código actual es correcto,
las imágenes regeneradas mostrarán dirección correcta y las guardadas no.
"""

import sys
import json
import math
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config
from src.ca.generate_dataset import generate_images_for_record, extract_weather_params
from src.data.preprocessing import DIR_VIENTO_CATS

_DIR_VIENTO_START = 14
_SQRT2 = math.sqrt(2) / 2.0
_DIR_TARGET = [
    ( 0.0,   +1.0  ),  # N
    (-_SQRT2,+_SQRT2), # NE
    (-1.0,    0.0  ),  # E
    (-_SQRT2,-_SQRT2), # SE
    ( 0.0,   -1.0  ),  # S
    (+_SQRT2,-_SQRT2), # SW
    (+1.0,    0.0  ),  # W
    (+_SQRT2,+_SQRT2), # NW
]

transform = T.Compose([T.Resize(64), T.ToTensor()])


def centroid_of_tensor(img_t):
    fire = img_t.mean(dim=0)  # [H, W]
    H, W = fire.shape
    yy = torch.arange(H, dtype=torch.float32) - (H - 1) / 2.0
    xx = torch.arange(W, dtype=torch.float32) - (W - 1) / 2.0
    mass = fire.sum().clamp(min=1e-6)
    cy = (fire * yy.view(H, 1)).sum() / mass
    cx = (fire * xx.view(1, W)).sum() / mass
    return cx.item(), cy.item()


def angular_error(cx, cy, tx, ty):
    if abs(cx) < 1e-6 and abs(cy) < 1e-6:
        return 180.0
    dot = cx * tx + cy * ty
    n = math.sqrt(cx**2 + cy**2) * math.sqrt(tx**2 + ty**2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / n))))


def main():
    cfg = get_config()
    condition_vectors = np.load(cfg.paths.condition_vectors)
    veg_profiles = np.load(cfg.paths.vegetation_profiles)

    with open(cfg.paths.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Encontrar registros E en val set
    val_dir = cfg.paths.ca_images / "val"
    with open(val_dir / "index.json") as f:
        val_index = json.load(f)

    wind_onehot_all = condition_vectors[:, _DIR_VIENTO_START:_DIR_VIENTO_START + 10]
    dir_idx_all = np.argmax(wind_onehot_all, axis=1)

    # Tomar 10 registros E únicos
    e_rec_indices = [e["record_idx"] for e in val_index
                     if dir_idx_all[e["record_idx"]] == 2]
    unique_e_recs = list(dict.fromkeys(e_rec_indices))[:10]

    print(f"Encontrados {len(unique_e_recs)} registros E únicos en val set")
    print(f"\n{'Rec':>6} {'--- GUARDADA ---':>20} {'--- REGENERADA ---':>20}")
    print(f"{'':>6} {'cx':>6} {'cy':>6} {'err':>6}  {'cx':>6} {'cy':>6} {'err':>6}")
    print("-" * 55)

    tx, ty = _DIR_TARGET[2]  # E target

    saved_errors = []
    regen_errors = []

    import tempfile, os
    tmpdir = Path(tempfile.mkdtemp())

    for rec_idx in unique_e_recs:
        # --- Imagen guardada ---
        saved_imgs = [e for e in val_index if e["record_idx"] == rec_idx]
        saved_cxs, saved_cys = [], []
        for entry in saved_imgs[:3]:
            img = Image.open(val_dir / entry["filename"]).convert("RGB")
            img_t = transform(img)
            cx, cy = centroid_of_tensor(img_t)
            saved_cxs.append(cx); saved_cys.append(cy)
        scx = np.mean(saved_cxs); scy = np.mean(saved_cys)
        serr = angular_error(scx, scy, tx, ty)
        saved_errors.append(serr)

        # --- Imagen regenerada ---
        entries = generate_images_for_record(
            record_idx=rec_idx,
            condition_vector=condition_vectors[rec_idx],
            veg_profile=veg_profiles[rec_idx],
            cfg=cfg,
            output_dir=tmpdir,
            base_seed=0,
            scaler=scaler,
        )
        regen_cxs, regen_cys = [], []
        for entry in entries[:3]:
            img = Image.open(tmpdir / entry["filename"]).convert("RGB")
            img_t = transform(img)
            cx, cy = centroid_of_tensor(img_t)
            regen_cxs.append(cx); regen_cys.append(cy)
        rcx = np.mean(regen_cxs); rcy = np.mean(regen_cys)
        rerr = angular_error(rcx, rcy, tx, ty)
        regen_errors.append(rerr)

        print(f"  {rec_idx:>4} {scx:>+6.1f} {scy:>+6.1f} {serr:>5.1f}°  "
              f"{rcx:>+6.1f} {rcy:>+6.1f} {rerr:>5.1f}°")

    print(f"\n  Media saved:  {np.mean(saved_errors):.1f}°")
    print(f"  Media regen:  {np.mean(regen_errors):.1f}°")

    # Limpiar
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()


if __name__ == "__main__":
    main()
