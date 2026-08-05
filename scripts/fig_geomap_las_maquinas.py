#!/usr/bin/env python3
"""Figura de validación geoespacial — LAS MÁQUINAS (Opción B).

Muestra las 6 realizaciones del generador Run 48 como heatmaps crudos 64×64
sin overlay satelital (que no funciona a escala de megaincendio). El fondo
oscuro y la flecha de dirección hacen legible la propagación hacia el norte.

Uso:
    python scripts/fig_geomap_las_maquinas.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import get_config
from src.models.generator import Generator
from scripts.veg_utils import composite_fire_on_veg, veg_legend_patches

CHECKPOINT = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
OUT        = Path("docs/figures/geomap_2940_LAS_MAQUINAS.png")
FIRE_IDX   = 2940

# Metadata del incendio (del CSV procesado)
FIRE_NAME   = "LAS MÁQUINAS"
FIRE_COMUNA = "Cauquenes"
FIRE_DATE   = "20/01/2017"
FIRE_SUP    = 159_812.6
FIRE_WIND   = "S"
FIRE_VEL    = 13.0
FIRE_TEMP   = 28.5
FIRE_LAT    = -35.8475
FIRE_LON    = -72.2817

N_SAMPLES = 6
BG  = "#111122"
TXT = "#FFFFFF"


def main():
    cfg = get_config()
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
    print(f"Checkpoint cargado (época {ckpt.get('epoch', '?')})")

    c = torch.tensor(cond_vecs[FIRE_IDX], dtype=torch.float32).unsqueeze(0).to(device)
    m = torch.tensor(masks_arr[FIRE_IDX],  dtype=torch.float32).unsqueeze(0).to(device)
    v = torch.tensor(veg_profs[FIRE_IDX],  dtype=torch.float32).unsqueeze(0).to(device)

    images = []
    for seed in range(N_SAMPLES):
        torch.manual_seed(seed + 42)
        z = torch.randn(1, cfg.model.z_dim, device=device)
        with torch.no_grad():
            img = gen(z, c, m, v)
        img_np = (img.squeeze().cpu().numpy() + 1) / 2
        images.append(np.transpose(img_np, (1, 2, 0)).clip(0, 1))

    # ── Figura 2×3 — landscape para slide 16:9 ───────────────────────────────
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.0 + 1.6),
                             gridspec_kw={"hspace": 0.26, "wspace": 0.08})
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        f"LAS MÁQUINAS — {FIRE_COMUNA}  ({FIRE_DATE},  {FIRE_SUP:,.0f} ha)  |  "
        f"Viento {FIRE_WIND}  {FIRE_VEL:.0f} km/h  |  T°: {FIRE_TEMP}°C\n"
        "Run 48 (época 70) — 6 realizaciones | "
        "Propagación esperada → NORTE  (downwind, viento del Sur)",
        fontsize=9.5, fontweight="bold", color=TXT, y=0.99,
    )

    h_img, w_img = 64, 64
    cx, cy = w_img // 2, h_img // 2
    # Flecha hacia arriba (Norte en imagen): de cy=32 a cy-16
    arrow_len = 14

    veg_profile = veg_profs[FIRE_IDX]

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(BG)
        if i >= N_SAMPLES:
            ax.axis("off")
            continue

        r = images[i][:, :, 0]
        g = images[i][:, :, 1]
        b = images[i][:, :, 2]
        fire = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
        vmax = max(fire.max() * 0.95, 0.05)
        composite = composite_fire_on_veg(fire / vmax, veg_profile)

        ax.imshow(composite, origin="upper", interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Realización {i + 1}", fontsize=8.5, color=TXT, pad=3)

        # Leyenda de vegetación solo en el primer panel
        if i == 0:
            patches = veg_legend_patches(veg_profile, top_n=3)
            if patches:
                ax.legend(handles=patches, loc="lower right", fontsize=6,
                          framealpha=0.60, facecolor="#111", edgecolor="none",
                          labelcolor="white", handlelength=1.0, borderpad=0.4,
                          title="Vegetación", title_fontsize=6)

        # Flecha cian apuntando al norte (arriba en imagen)
        ax.annotate(
            "", xy=(cx, cy - arrow_len), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2.2,
                            mutation_scale=14),
            zorder=5,
        )
        ax.text(cx, cy - arrow_len - 2, "N", color="cyan",
                fontsize=8, fontweight="bold", ha="center", va="bottom",
                zorder=5)

        # Punto de ignición
        ax.plot(cx, cy, "r*", markersize=8, zorder=6)

    # Nota de escala y limitación
    fig.text(
        0.5, 0.01,
        "Grid CA: 64×64 celdas | Escala ≈ 12,8 km × 12,8 km  (200 m/celda, clampeado al máximo)\n"
        "Overlay satelital omitido: el área quemada real (lon ≈ −72,80°) no se solapa con el "
        "grid GAN (lon ≈ −72,28°) por ser un megaincendio fuera del rango de escala del modelo.",
        ha="center", fontsize=7, color="#AAAACC", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1A1A33", ec="#444466", alpha=0.85),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
