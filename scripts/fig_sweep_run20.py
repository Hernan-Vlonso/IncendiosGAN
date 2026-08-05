#!/usr/bin/env python3
"""Sweep de dirección de viento — Run 20 (versión limpia para memoria).

Genera 9 imágenes (N, NE, E, SE, S, SW, W, NW, Calma) con el mismo z
y distinta Dir_viento, usando colormap inferno para máxima legibilidad.

Salida: docs/figures/sweep_wind_direction_run20.png

Uso:
    python scripts/fig_sweep_run20.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.config import get_config  # noqa: E402 (sys.path ya modificado)

# Run 20 usa checkpoint en el root del proyecto
CHECKPOINT = Path("checkpoint_best_run20.pt")
OUT        = Path("docs/figures/sweep_wind_direction_run20.png")


# ── Arquitectura Run 20 (pre-CBN, BN estándar) ────────────────────────────────
# Inferida de los keys del checkpoint:
#   cond_embedding.net.{0,2,4} → MLP(74→256→256→128)
#   project.{0,1}              → Linear(256,8192) + BN1d(8192)
#   blocks.{0,1,2}.{0,1}       → ConvTranspose2d + BN2d
#   final.0                    → ConvTranspose2d(64,3,4,2,1)
class _CondEmb(nn.Module):
    def __init__(self, in_dim=74, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256),    nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class GeneratorRun20(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.cond_embedding = _CondEmb(74, 128)
        self.project = nn.Sequential(
            nn.Linear(256, 8192),
            nn.BatchNorm1d(8192),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256)),
            nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128)),
            nn.Sequential(nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False), nn.BatchNorm2d(64)),
        ])
        self.final = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False))

    def forward(self, z, cond_cat):
        e = self.cond_embedding(cond_cat)
        x = torch.relu(self.project(torch.cat([z, e], dim=1)))
        x = x.view(-1, 512, 4, 4)
        for block in self.blocks:
            x = torch.relu(block(x))
        return torch.tanh(self.final(x))

DIR_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
# Downwind (propagación esperada) para cada dirección — para la flecha
PROPAGATION = {
    "N":    ( 0,  1), "NE": (-1,  1), "E":  (-1,  0), "SE": (-1, -1),
    "S":    ( 0, -1), "SW": ( 1, -1), "W":  ( 1,  0), "NW": ( 1,  1),
    "Calma": None,
}
DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma"]

BG  = "#111122"
TXT = "#FFFFFF"


def fire_intensity(img_rgb):
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    return np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)


def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cond_vecs = np.load(cfg.paths.condition_vectors)
    veg_profs = np.load(cfg.paths.vegetation_profiles)
    masks_arr = np.load(cfg.paths.masks)

    gen = GeneratorRun20(z_dim=128).to(device)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    gen.load_state_dict(ckpt.get("ema", ckpt.get("generator")))
    gen.eval()
    print(f"Run 20 checkpoint cargado (época {ckpt.get('epoch', '?')})")

    # Incendio base con viento conocido (condiciones base fijas para el sweep)
    base_idx = 2940   # LAS MAQUINAS — buenas condiciones base
    base_cond = cond_vecs[base_idx].copy().astype(np.float32)   # 32 dims
    mask_v    = masks_arr[base_idx].copy().astype(np.float32)   # 32 dims
    veg       = veg_profs[base_idx].astype(np.float32)          # 10 dims

    torch.manual_seed(42)
    z_fixed = torch.randn(1, 128, device=device)

    images = {}
    with torch.no_grad():
        for wind_dir in DIRS:
            cond = base_cond.copy()
            # Reemplazar one-hot Dir_viento (dims 14-23)
            for i in range(10):
                cond[14 + i] = 0.0
            cond[14 + DIR_CATS.index(wind_dir)] = 1.0

            # Run 20: cat(condition, mask, veg) → 74 dims
            cond_cat = np.concatenate([cond, mask_v, veg])
            c = torch.tensor(cond_cat, dtype=torch.float32).unsqueeze(0).to(device)

            img = gen(z_fixed, c)
            img = (img.squeeze().cpu().numpy() + 1) / 2
            images[wind_dir] = np.transpose(img, (1, 2, 0)).clip(0, 1)

    # ── Figura 1×9 ────────────────────────────────────────────────────────────
    n    = len(DIRS)
    cell = 2.5
    fig, axes = plt.subplots(1, n, figsize=(n * cell, cell + 1.4),
                              gridspec_kw={"wspace": 0.06})
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        "Sweep Dir_viento — Run 20 (condiciones base fijas, mismo $z$)\n"
        "Viento DESDE la dirección indicada; fuego se propaga en dirección opuesta (downwind)",
        fontsize=9, fontweight="bold", color=TXT, y=1.02,
    )

    for ax, wind_dir in zip(axes, DIRS):
        ax.set_facecolor(BG)
        fire = fire_intensity(images[wind_dir])
        ax.imshow(fire, cmap="inferno", origin="upper",
                  interpolation="bilinear", vmin=0.0, vmax=1.0)
        ax.axis("off")
        ax.set_title(wind_dir, fontsize=9, fontweight="bold", color=TXT, pad=3)

        # Flecha de propagación esperada
        prop = PROPAGATION.get(wind_dir)
        if prop is not None:
            dx, dy = prop
            H, W = fire.shape
            cx, cy = W / 2, H / 2
            scale  = H * 0.26
            ax.annotate(
                "", xy=(cx + dx * scale, cy + dy * scale), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="cyan",
                                lw=2.0, mutation_scale=12),
                zorder=5,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
