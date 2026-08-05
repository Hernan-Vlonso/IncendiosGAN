#!/usr/bin/env python3
"""Figura comparación Real (CA) vs Generado (cGAN) — Run 48.

Dos filas × 8 columnas (una por dirección de viento).
Usa extracción de canal de fuego + colormap inferno para
garantizar contraste visual limpio en ambas filas.

Salida: docs/figures/comparacion_real_vs_generado_run48.png

Uso:
    python scripts/fig_comparison_run48.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from PIL import Image

from src.config import get_config
from src.models.generator import Generator

CHECKPOINT = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
OUT = Path("docs/figures/comparacion_real_vs_generado_run48.png")

# 4 direcciones cardinales — paneles grandes para que sean legibles en proyección.
# Las 8 direcciones ya se muestran en el slide de validación de condicionamiento.
CASES = [
    ("N",  5223, "Constitución 2"),
    ("E",  1417, "Cnel. Maule"),
    ("S",  5269, "Santo Toribio"),
    ("W",  1189, "El Ajial"),
]

BG   = "#111122"
TXT  = "#EEEEEE"
GRIS = "#888888"


def extract_fire(img_rgb: np.ndarray, smooth: float = 1.2) -> np.ndarray:
    """Extrae intensidad de fuego, elimina fondo y suaviza artefactos CBN.

    El modelo CBN imprime un patrón reticular (~2 px de período) en toda
    la imagen. El suavizado gaussiano (sigma=1.2) promedia ese ruido de
    alta frecuencia sin borrar el blob de fuego (~20-40 px de diámetro).
    Se aplica igual a CA y GAN para que ambas filas sean comparables.
    """
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    raw    = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
    smooth = gaussian_filter(raw, sigma=smooth)   # elimina artefacto CBN
    bg     = np.percentile(smooth, 8)             # nivel de fondo
    return np.maximum(smooth - bg, 0)


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

    cmap_fire = plt.get_cmap("inferno")

    def to_heatmap(img_rgb: np.ndarray) -> np.ndarray:
        """Extrae fuego y renderiza con inferno sobre negro — igual para CA y GAN."""
        fire = extract_fire(img_rgb)
        vmax = max(fire.max() * 0.95, 0.05)
        fire_norm = np.clip(fire / vmax, 0, 1)
        return cmap_fire(fire_norm)[:, :, :3].astype(np.float32)

    N = len(CASES)
    real_comp, gen_comp = [], []

    for wind_dir, idx, name in CASES:
        vp = veg_profs[idx]

        # Imagen CA real — último horizonte (t=80)
        h_files = sorted((cfg.paths.ca_images / "val").glob(
            f"rec{idx:05d}_var0_h*.png"))
        if not h_files:
            h_files = sorted((cfg.paths.ca_images / "train").glob(
                f"rec{idx:05d}_var0_h*.png"))
        if h_files:
            img_ca = np.array(Image.open(h_files[-1]).convert("RGB")) / 255.0
            print(f"  {wind_dir:3s} [{name:15s}] CA ok — fire_max={extract_fire(img_ca).max():.3f}")
        else:
            print(f"  {wind_dir:3s} [{name:15s}] CA NO ENCONTRADO")
            img_ca = np.zeros((64, 64, 3))
        real_comp.append(to_heatmap(img_ca))

        # Imagen generada
        torch.manual_seed(42)
        c = torch.tensor(cond_vecs[idx], dtype=torch.float32).unsqueeze(0).to(device)
        m = torch.tensor(masks_arr[idx], dtype=torch.float32).unsqueeze(0).to(device)
        v = torch.tensor(vp, dtype=torch.float32).unsqueeze(0).to(device)
        z = torch.randn(1, cfg.model.z_dim, device=device)
        with torch.no_grad():
            img_gen = gen(z, c, m, v)
        img_np = np.transpose((img_gen.squeeze().cpu().numpy() + 1) / 2, (1, 2, 0)).clip(0, 1)
        gen_comp.append(to_heatmap(img_np))

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, N, figsize=(N * 3.6, 7.5),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.06})
    fig.patch.set_facecolor(BG)

    for i, (wind_dir, _, name) in enumerate(CASES):
        for row, comp in enumerate([real_comp[i], gen_comp[i]]):
            ax = axes[row, i]
            ax.imshow(comp, origin="upper", interpolation="bilinear")
            ax.set_facecolor(BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        axes[0, i].set_title(f"Viento {wind_dir}", fontsize=15, color=TXT,
                              pad=7, fontweight="bold")
        axes[1, i].set_xlabel(name, fontsize=10, color=GRIS, labelpad=4)

    # Etiquetas de fila
    axes[0, 0].set_ylabel("Simulación CA\n(t = 80)", fontsize=13,
                           fontweight="bold", color=TXT, labelpad=10)
    axes[1, 0].set_ylabel("cGAN generado\n(Run 48)", fontsize=13,
                           fontweight="bold", color=TXT, labelpad=10)

    plt.subplots_adjust(top=0.88, bottom=0.07, left=0.12, right=0.99,
                        hspace=0.08, wspace=0.06)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nGuardado: {OUT}")


if __name__ == "__main__":
    main()
