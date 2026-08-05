#!/usr/bin/env python3
"""Diagnóstico 5: ¿El generador usa la condición?

Fija z, varía solo la dirección de viento en el condition vector,
mide si los outputs cambian y si el centroide apunta en dirección correcta.
Usa checkpoint_best_dir.pt de run34 (el mejor checkpoint conocido).
"""

import sys
import math
import glob
from pathlib import Path
import numpy as np
import torch
import torchvision.utils as vutils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config
from src.training.trainer import Trainer

_SQRT2 = math.sqrt(2) / 2.0
_DIRS  = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_DIR_TARGET = [
    ( 0.0,   +1.0  ),
    (-_SQRT2,+_SQRT2),
    (-1.0,    0.0  ),
    (-_SQRT2,-_SQRT2),
    ( 0.0,   -1.0  ),
    (+_SQRT2,-_SQRT2),
    (+1.0,    0.0  ),
    (+_SQRT2,+_SQRT2),
]
_DIR_VIENTO_START = 14


def angular_error(cx, cy, tx, ty):
    if abs(cx) < 1e-6 and abs(cy) < 1e-6:
        return 180.0
    dot = cx * tx + cy * ty
    n = math.sqrt(cx**2 + cy**2) * math.sqrt(tx**2 + ty**2)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / n))))


def find_best_checkpoint():
    """Buscar el mejor checkpoint disponible."""
    candidates = [
        "outputs/checkpoints/run34/checkpoint_best_dir.pt",
        "outputs/checkpoints/run34/checkpoint_epoch0020.pt",
        "outputs/checkpoints/run38/checkpoint_best_dir.pt",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    # Buscar cualquier checkpoint_best_dir
    found = glob.glob("outputs/checkpoints/**/checkpoint_best_dir.pt", recursive=True)
    if found:
        return found[0]
    found = glob.glob("outputs/checkpoints/**/*.pt", recursive=True)
    return found[0] if found else None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint if args.checkpoint else find_best_checkpoint()
    if ckpt_path is None:
        print("ERROR: No se encontró ningún checkpoint.")
        return
    print(f"Usando checkpoint: {ckpt_path}")

    # Cargar modelo
    trainer = Trainer(cfg, None, None, [])
    trainer.load_checkpoint(ckpt_path)
    trainer.ema.shadow.eval()

    # Cargar un batch real del val set para tener conditions realistas
    condition_vectors = np.load(cfg.paths.condition_vectors)
    veg_profiles = np.load(cfg.paths.vegetation_profiles)
    masks = np.load(cfg.paths.masks)

    # Tomar un registro base con condiciones medianas (S wind, índice arbitrario)
    wind_onehot_all = condition_vectors[:, _DIR_VIENTO_START:_DIR_VIENTO_START + 10]
    dir_idx_all = np.argmax(wind_onehot_all, axis=1)
    s_indices = np.where(dir_idx_all == 4)[0][:1]  # 1 registro S como base
    base_idx = int(s_indices[0])

    base_cond = condition_vectors[base_idx].copy()
    base_mask = masks[base_idx].copy()
    base_veg  = veg_profiles[base_idx].copy()

    # N muestras por dirección
    N = 16
    torch.manual_seed(42)
    fixed_z = torch.randn(N, cfg.model.z_dim, device=device)

    print(f"\nBase record: {base_idx}")
    print(f"N samples per direction: {N}\n")
    print(f"{'Dir':>4} {'cx_mean':>9} {'cy_mean':>9} {'disp':>7} {'error':>8} {'OK?':>5}")
    print("-" * 50)

    results = {}
    imgs_all = []

    for d, name in enumerate(_DIRS):
        # Construir condition vector con solo Dir_viento=d activo
        cond = base_cond.copy()
        # Limpiar todo el bloque Dir_viento (posiciones 14-23)
        cond[14:24] = 0.0
        # Activar dirección d
        cond[14 + d] = 1.0

        cond_t = torch.from_numpy(cond).float().unsqueeze(0).expand(N, -1).to(device)
        mask_t = torch.from_numpy(base_mask).float().unsqueeze(0).expand(N, -1).to(device)
        veg_t  = torch.from_numpy(base_veg).float().unsqueeze(0).expand(N, -1).to(device)

        with torch.no_grad():
            fake = trainer.ema.shadow(fixed_z, cond_t, mask_t, veg_t)

        # Centroid en [-1,1] → [0,1]
        fire = (fake.mean(dim=1) + 1.0) / 2.0  # [N, H, W]
        B, H, W = fire.shape
        yy = torch.arange(H, dtype=torch.float32, device=device) - (H-1)/2.0
        xx = torch.arange(W, dtype=torch.float32, device=device) - (W-1)/2.0
        mass = fire.sum(dim=(1,2)).clamp(min=1e-6)
        cy = (fire * yy.view(1,H,1)).sum(dim=(1,2)) / mass
        cx = (fire * xx.view(1,1,W)).sum(dim=(1,2)) / mass

        mean_cx = cx.mean().item()
        mean_cy = cy.mean().item()
        disp = math.sqrt(mean_cx**2 + mean_cy**2)

        tx, ty = _DIR_TARGET[d]
        err = angular_error(mean_cx, mean_cy, tx, ty)
        ok = "OK" if err < 45 else "FAIL"

        print(f"  {name:>4} {mean_cx:>+7.2f}   {mean_cy:>+7.2f}   {disp:>5.2f}   {err:>6.1f}°  {ok}")
        results[name] = {"cx": mean_cx, "cy": mean_cy, "err": err}
        imgs_all.append(fake[:4])  # primeras 4 por dir

    # Guardar grilla visual: 8 dirs × 4 imágenes
    grid_imgs = torch.cat(imgs_all, dim=0)  # [32, C, H, W]
    grid_imgs = (grid_imgs + 1) / 2  # [-1,1] → [0,1]
    grid = vutils.make_grid(grid_imgs, nrow=4, normalize=False, padding=2)
    out = cfg.paths.figures / "diag05_generator_conditioning.png"
    vutils.save_image(grid, str(out))

    n_ok = sum(1 for v in results.values() if v["err"] < 45)
    mean_err = np.mean([v["err"] for v in results.values()])
    print(f"\n  Correctas (<45°): {n_ok}/8   Error medio: {mean_err:.1f}°")
    print(f"  Grid guardado: {out}")

    # Test de sensibilidad: ¿cambia el output cuando cambia la dirección?
    print(f"\n  Sensibilidad (std del centroide entre direcciones):")
    all_cx = [results[n]["cx"] for n in _DIRS]
    all_cy = [results[n]["cy"] for n in _DIRS]
    print(f"    std(cx) = {np.std(all_cx):.3f}   std(cy) = {np.std(all_cy):.3f}")
    print(f"    (0 = sin sensibilidad, >1 = buena respuesta a dirección)")


if __name__ == "__main__":
    main()
