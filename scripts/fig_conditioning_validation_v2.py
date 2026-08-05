#!/usr/bin/env python3
"""Figura de validación de condicionamiento — versión mejorada para memoria/defensa.

Usa colormap térmico (inferno) sobre el canal de intensidad de fuego en lugar
de RGB crudo, para que la dirección de propagación sea claramente visible.

Salida: docs/figures/fig_conditioning_validation.png

Uso:
    python scripts/fig_conditioning_validation_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

from src.config import get_config
from src.models.generator import Generator
from scripts.veg_utils import composite_fire_on_veg, veg_legend_patches

CHECKPOINT = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
OUT        = Path("docs/figures/fig_conditioning_validation.png")

CASES = [
    ("N",  5247, "BATUCO"),
    ("NE", 2915, "TABUNCO"),
    ("E",  2939, "SANTA CRUZ"),
    ("SE", 2956, "CORONEL DE MAULE"),
    ("S",  2940, "LAS MAQUINAS"),
    ("SW", 5249, "QUIVOLGO"),
    ("W",  2992, "LAS CARDILLAS"),
    ("NW", 4932, "CORRAL DE PEREZ"),
]
N_SAMPLES = 1

# Dirección de propagación en coordenadas de imagen (fila 0 = arriba = Norte)
# dx = columna (+ = E), dy = fila (- = N / arriba)
PROPAGATION = {
    "N":  ( 0,  1), "NE": (-1,  1), "E":  (-1,  0), "SE": (-1, -1),
    "S":  ( 0, -1), "SW": ( 1, -1), "W":  ( 1,  0), "NW": ( 1,  1),
}
COMPASS = {
    "N": "→S", "NE": "→SW", "E": "→W",  "SE": "→NW",
    "S": "→N", "SW": "→NE", "W": "→E",  "NW": "→SE",
}

BG  = "#111122"
TXT = "#FFFFFF"


def fire_intensity(img_rgb: np.ndarray, smooth: float = 1.2) -> np.ndarray:
    """Canal escalar de intensidad de fuego con remoción de fondo CBN."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    raw = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
    sm  = gaussian_filter(raw, sigma=smooth)
    bg  = np.percentile(sm, 8)
    return np.maximum(sm - bg, 0)


def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cond_vecs = np.load(cfg.paths.condition_vectors)
    veg_profs = np.load(cfg.paths.vegetation_profiles)
    masks_arr = np.load(cfg.paths.masks)

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
    print(f"Run 48 época {ckpt.get('epoch', '?')} cargado.")

    def generate(idx, seed=42):
        torch.manual_seed(seed)
        c = torch.tensor(cond_vecs[idx], dtype=torch.float32).unsqueeze(0).to(device)
        m = torch.tensor(masks_arr[idx], dtype=torch.float32).unsqueeze(0).to(device)
        v = torch.tensor(veg_profs[idx], dtype=torch.float32).unsqueeze(0).to(device)
        imgs = []
        with torch.no_grad():
            for _ in range(N_SAMPLES):
                z = torch.randn(1, cfg.model.z_dim, device=device)
                img = gen(z, c, m, v)
                img = (img.squeeze().cpu().numpy() + 1) / 2
                imgs.append(np.transpose(img, (1, 2, 0)).clip(0, 1))
        return imgs

    # ── Figura: layout landscape 2 filas × 4 cols ─────────────────────────────
    # Fila 0: N, NE, E, SE  |  Fila 1: S, SW, W, NW
    GRID = [CASES[:4], CASES[4:]]   # 2 grupos de 4 direcciones
    NCOLS, NROWS = 4, 2

    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(NCOLS * 3.2, NROWS * 3.2 + 0.6),
        gridspec_kw={"hspace": 0.14, "wspace": 0.08},
    )
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        "Flecha cian = dirección downwind esperada  |  Colormap: intensidad de fuego",
        fontsize=9.5, color="#AAAACC", y=1.005,
    )

    for r, group in enumerate(GRID):
        for c, (wind_dir, idx, fire_name) in enumerate(group):
            imgs = generate(idx, seed=42)
            dx, dy = PROPAGATION[wind_dir]
            ax = axes[r][c]
            ax.set_facecolor(BG)

            fire = fire_intensity(imgs[0])
            vmax = max(fire.max() * 0.95, 0.05)
            fire_norm = np.clip(fire / vmax, 0, 1)

            composite = composite_fire_on_veg(fire_norm, veg_profs[idx])
            ax.imshow(composite, origin="upper", interpolation="bilinear")
            ax.axis("off")

            # Leyenda vegetación dominante (top 2)
            patches = veg_legend_patches(veg_profs[idx], top_n=2)
            if patches:
                ax.legend(handles=patches, loc="lower right", fontsize=5.5,
                          framealpha=0.55, facecolor="#111", edgecolor="none",
                          labelcolor="white", handlelength=1.0, borderpad=0.4)

            # Título: dirección + nombre
            ax.set_title(f"Viento {wind_dir}  ({COMPASS[wind_dir]})",
                         fontsize=11, fontweight="bold", color=TXT, pad=5)
            ax.text(0.5, -0.04, fire_name, ha="center", va="top",
                    fontsize=8, color="#AAAACC", transform=ax.transAxes)

            # Flecha de propagación esperada
            H, W = fire.shape
            cx, cy = W / 2, H / 2
            scale  = H * 0.28
            ax.annotate(
                "", xy=(cx + dx * scale, cy + dy * scale), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="cyan",
                                lw=3.0, mutation_scale=20),
                zorder=5,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
