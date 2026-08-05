#!/usr/bin/env python3
"""Figura para defensa: condicionamiento direccional sobre caso real — EL BOLDO.

Dos paneles:
  Izquierdo: 6 realizaciones individuales (mosaico 2×3) mostrando variabilidad estocástica.
  Derecho:   heatmap promedio con flecha cian del centroide (dirección downwind).

El propósito no es comparar con el área quemada real (incompatible en escala/resolución),
sino demostrar que el condicionamiento meteorológico produce la orientación correcta
(viento Sur → propagación Norte) sobre datos CONAF reales.

Salida: docs/figures/geomap_real_case_el_boldo.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from src.config import get_config
from src.models.generator import Generator
from scripts.veg_utils import composite_fire_on_veg, veg_legend_patches

CHECKPOINT  = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
RECORD_IDX  = 1653          # EL BOLDO — 376 ha, Dir=S, 2019-01-02
N_SAMPLES   = 6
OUT         = Path("docs/figures/geomap_real_case_el_boldo.png")

BG  = "#0d0d1a"
TXT = "#FFFFFF"
ACC = "#AAAACC"


def fire_intensity(arr):
    """Canal rojo dominante como proxy de intensidad de fuego."""
    r, g, b = arr[0], arr[1], arr[2]
    return np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)


def main():
    cfg    = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cond_vecs = np.load(cfg.paths.condition_vectors)
    veg_profs = np.load(cfg.paths.vegetation_profiles)
    masks_arr = np.load(cfg.paths.masks)

    gen = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        img_channels=cfg.model.img_channels,
        features=cfg.model.g_features,
        veg_dim=cfg.model.veg_dim,
    ).to(device)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    gen.load_state_dict(ckpt.get("ema", ckpt.get("generator")))
    gen.eval()

    c = torch.tensor(cond_vecs[RECORD_IDX], dtype=torch.float32).unsqueeze(0).to(device)
    m = torch.tensor(masks_arr[RECORD_IDX],  dtype=torch.float32).unsqueeze(0).to(device)
    v = torch.tensor(veg_profs[RECORD_IDX],  dtype=torch.float32).unsqueeze(0).to(device)

    # Generar N_SAMPLES realizaciones
    raw_imgs  = []   # RGB [0,1] para el mosaico
    fire_maps = []   # escalar de intensidad para el heatmap

    for seed in range(N_SAMPLES):
        torch.manual_seed(seed + 42)
        z = torch.randn(1, cfg.model.z_dim, device=device)
        with torch.no_grad():
            img = gen(z, c, m, v)
        arr = (img.squeeze().cpu().numpy() + 1) / 2          # (3, H, W) [0,1]
        raw_imgs.append(np.transpose(arr, (1, 2, 0)))        # (H, W, 3)
        fire_maps.append(fire_intensity(arr))

    heatmap   = np.mean(fire_maps, axis=0)                   # (H, W) promedio
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_n = (heatmap - h_min) / max(h_max - h_min, 1e-6)

    # Centroide ponderado para flecha direccional
    H, W  = heatmap_n.shape
    yy    = np.arange(H) - (H - 1) / 2.0
    xx    = np.arange(W) - (W - 1) / 2.0
    mass  = heatmap_n.sum().clip(min=1e-6)
    cy    = (heatmap_n * yy[:, None]).sum() / mass
    cx    = (heatmap_n * xx[None, :]).sum() / mass
    norm  = max(np.sqrt(cx**2 + cy**2), 1e-6)
    scale = (H / 2) * 0.55
    ax_dy = cy / norm * scale
    ax_dx = cx / norm * scale

    veg_profile = veg_profs[RECORD_IDX]

    # Compositar realizaciones individuales con fondo vegetación
    # Gaussian smoothing elimina el artefacto de cuadrícula CBN (~2 px de período)
    composite_imgs = []
    for img_rgb in raw_imgs:
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        fire_raw = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
        fire_raw = gaussian_filter(fire_raw, sigma=1.2)
        bg_level = np.percentile(fire_raw, 8)
        fire_raw = np.maximum(fire_raw - bg_level, 0)
        vmax_i = max(fire_raw.max() * 0.95, 0.05)
        composite_imgs.append(composite_fire_on_veg(fire_raw / vmax_i, veg_profile))

    # También suavizar el heatmap promedio antes de compositar
    heatmap_n = gaussian_filter(heatmap_n, sigma=1.0)

    # ── Figura ───────────────────────────────────────────────────────────────
    # Layout: mosaico 2×3 + heatmap grande
    fig = plt.figure(figsize=(13, 5.5), facecolor=BG)
    gs  = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1, 1.15],
        wspace=0.06,
        left=0.02, right=0.98, top=0.84, bottom=0.04
    )

    # ── Panel izquierdo: mosaico 2×3 de realizaciones individuales ──────────
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0], hspace=0.04, wspace=0.04)
    for i in range(N_SAMPLES):
        ax = fig.add_subplot(gs_left[i // 3, i % 3])
        ax.imshow(composite_imgs[i], interpolation="bilinear")
        ax.axis("off")
        ax.set_facecolor(BG)
        ax.text(0.97, 0.03, f"#{i+1}", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=7, color=ACC)

    # Título del mosaico
    fig.text(0.255, 0.87, "6 realizaciones individuales (variabilidad estocástica)",
             ha="center", va="bottom", fontsize=9, color=TXT)

    # ── Panel derecho: heatmap promedio con flecha ───────────────────────────
    ax_r = fig.add_subplot(gs[1])
    ax_r.set_facecolor(BG)
    ax_r.axis("off")

    composite_avg = composite_fire_on_veg(heatmap_n, veg_profile)
    im = ax_r.imshow(composite_avg, interpolation="bilinear")

    cx_c, cy_c = W / 2.0, H / 2.0
    ax_r.annotate(
        "", xy=(cx_c + ax_dx, cy_c + ax_dy), xytext=(cx_c, cy_c),
        arrowprops=dict(arrowstyle="-|>", color="cyan", lw=2.8, mutation_scale=18),
        zorder=5
    )
    ax_r.plot(cx_c, cy_c, "c+", markersize=9, markeredgewidth=2, zorder=6)

    # Rosa de los vientos mínima
    for label, (dx2, dy2) in [("N", (0, -1)), ("S", (0, 1)), ("E", (1, 0)), ("O", (-1, 0))]:
        ax_r.text(3 + dx2 * 5.5, 3 + dy2 * 5.5, label, color=ACC,
                  fontsize=7, ha="center", va="center", zorder=7)

    ax_r.set_title(
        "Heatmap promedio (6 realizaciones)\n"
        "Flecha cian = centroide de propagación (apunta al Norte)\n"
        "Dir_viento=S  →  propagación downwind al Norte",
        fontsize=9, color=TXT, pad=5
    )

    # Leyenda de vegetación
    patches = veg_legend_patches(veg_profile, top_n=3)
    if patches:
        ax_r.legend(handles=patches, loc="lower right", fontsize=7,
                    framealpha=0.60, facecolor="#111", edgecolor="none",
                    labelcolor="white", handlelength=1.2, borderpad=0.5,
                    title="Vegetación", title_fontsize=7)

    fig.suptitle(
        "Condicionamiento direccional sobre caso real — EL BOLDO, Maule\n"
        "(376 ha, viento Sur, 28 °C, HR 31%, 2019-01-02)  ·  Run 48 (época 70)",
        fontsize=10, fontweight="bold", color=TXT, y=0.99
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
