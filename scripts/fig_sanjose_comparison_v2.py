#!/usr/bin/env python3
"""Comparación SAN JOSE: Dir_viento=S (conocido) vs Dir_viento=Missing — versión mejorada.

Usa colormap térmico para mostrar con claridad la diferencia direccional
entre tener o no tener dirección de viento disponible.

Salida: docs/figures/comparison_sanjose_complete_vs_incomplete.png

Uso:
    python scripts/fig_sanjose_comparison_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

from src.config import get_config
from src.models.generator import Generator
from scripts.veg_utils import composite_fire_on_veg, veg_legend_patches

CHECKPOINT = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
OUT        = Path("docs/figures/comparison_sanjose_complete_vs_incomplete.png")

DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
FIRE_IDX        = 4   # SAN JOSE — 12 ha, Viento Sur 9,8 km/h
N_REAL          = 2

BG  = "#111122"
TXT = "#FFFFFF"


def fire_intensity(img_rgb: np.ndarray) -> np.ndarray:
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    raw = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
    sm  = gaussian_filter(raw, sigma=1.2)
    bg  = np.percentile(sm, 8)
    return np.maximum(sm - bg, 0)


def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cond_vecs = np.load(cfg.paths.condition_vectors)
    veg_profs = np.load(cfg.paths.vegetation_profiles)
    masks_arr = np.load(cfg.paths.masks)

    cond_s = cond_vecs[FIRE_IDX].copy().astype(np.float32)
    mask_v = masks_arr[FIRE_IDX].copy().astype(np.float32)
    veg    = veg_profs[FIRE_IDX].astype(np.float32)

    # Variante Missing: reemplazar one-hot Dir_viento
    cond_m = cond_s.copy()
    for i in range(10):
        cond_m[14 + i] = 0.0
    cond_m[14 + DIR_VIENTO_CATS.index("Missing")] = 1.0

    gen = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        veg_dim=cfg.model.veg_dim,
    ).to(device)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    gen.load_state_dict(ckpt.get("ema", ckpt.get("generator")))
    gen.eval()
    print(f"Generador cargado (epoch {ckpt.get('epoch', '?')})")

    def generate(cond, n=N_REAL, seed=42):
        torch.manual_seed(seed)
        c = torch.tensor(cond,  dtype=torch.float32).unsqueeze(0).to(device)
        m = torch.tensor(mask_v, dtype=torch.float32).unsqueeze(0).to(device)
        v = torch.tensor(veg,    dtype=torch.float32).unsqueeze(0).to(device)
        imgs = []
        with torch.no_grad():
            for _ in range(n):
                z = torch.randn(1, cfg.model.z_dim, device=device)
                img = gen(z, c, m, v)
                img = (img.squeeze().cpu().numpy() + 1) / 2
                imgs.append(np.transpose(img, (1, 2, 0)).clip(0, 1))
        return imgs

    imgs_s = generate(cond_s)
    imgs_m = generate(cond_m)

    # ── Figura: 2 filas × 2 cols — landscape para caber en columna del slide ──
    fig, axes = plt.subplots(
        N_REAL, 2,
        figsize=(9.0, 5.5),
        gridspec_kw={"hspace": 0.10, "wspace": 0.08},
    )
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        "SAN JOSE — Viento Sur  |  Completo vs. Dir=Missing",
        fontsize=10, color="#AAAACC", y=1.02,
    )

    col_titles = [
        "Dir_viento = S  (conocido)\nPropagación → NORTE",
        "Dir_viento = Missing\n(resto idéntico)",
    ]
    col_colors = ["#00E5FF", "#FF9900"]

    for j in range(2):
        axes[0][j].set_title(col_titles[j], fontsize=10.5, fontweight="bold",
                             color=col_colors[j], pad=7)

    for i in range(N_REAL):
        for j, imgs in enumerate([imgs_s, imgs_m]):
            ax = axes[i][j]
            ax.set_facecolor(BG)
            fire = fire_intensity(imgs[i])
            vmax = max(fire.max() * 0.95, 0.05)
            fire_norm = np.clip(fire / vmax, 0, 1)
            composite = composite_fire_on_veg(fire_norm, veg)
            ax.imshow(composite, origin="upper", interpolation="bilinear")
            ax.axis("off")

            H, W = fire.shape
            cx, cy = W / 2, H / 2

            if j == 0:
                ax.annotate(
                    "", xy=(cx, cy - 15), xytext=(cx, cy + 4),
                    arrowprops=dict(arrowstyle="-|>", color="cyan",
                                    lw=2.5, mutation_scale=16),
                    zorder=5,
                )
                ax.text(cx, cy - 19, "N", color="cyan", fontsize=9,
                        fontweight="bold", ha="center", va="bottom", zorder=5)
            else:
                ax.text(cx, cy - H * 0.28, "?", color="#FF9900", fontsize=20,
                        fontweight="bold", ha="center", va="center",
                        alpha=0.85, zorder=5)

            ax.text(3, H - 4, f"R{i+1}", color="white", fontsize=8,
                    va="bottom", ha="left", zorder=5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
