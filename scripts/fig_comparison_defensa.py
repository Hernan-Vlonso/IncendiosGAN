#!/usr/bin/env python3
"""Figura de comparación Real (CA) vs Generado para la defensa.

Usa los 8 incendios representativos de make_conditioning_validation.py
(incendios grandes, una realización cada uno) para que el contraste
visual sea claro. Muestra también el canal rojo (intensidad) en escala
de calor para mayor claridad en proyector.

Uso:
    python scripts/fig_comparison_defensa.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from src.config import get_config
from src.models.generator import Generator
from scripts.veg_utils import composite_fire_on_veg

CHECKPOINT = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
OUT = Path("docs/figures/fig_comparison_defensa.png")

# Mismo set que conditioning_validation — incendios grandes y representativos
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

BG  = "#FAFAFA"
TXT = "#1A1A2E"


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

    N = len(CASES)
    real_imgs, gen_imgs, titles = [], [], []

    def to_fire_composite(rgb_hwc: np.ndarray, veg_profile: np.ndarray) -> np.ndarray:
        r, g, b = rgb_hwc[:, :, 0], rgb_hwc[:, :, 1], rgb_hwc[:, :, 2]
        fire = np.clip((r - g * 0.5 - b * 0.2) * 2.5 + 0.15, 0, 1)
        vmax = max(fire.max() * 0.95, 0.05)
        return composite_fire_on_veg(fire / vmax, veg_profile)

    for wind_dir, idx, name in CASES:
        vp = veg_profs[idx]

        # Imagen real CA (último horizonte t=80)
        h_files = sorted((cfg.paths.ca_images / "val").glob(f"rec{idx:05d}_var0_h*.png"))
        if not h_files:
            h_files = sorted((cfg.paths.ca_images / "train").glob(f"rec{idx:05d}_var0_h*.png"))
        if h_files:
            raw_real = np.array(Image.open(h_files[-1]).convert("RGB")) / 255.0
            real_imgs.append(to_fire_composite(raw_real, vp))
        else:
            real_imgs.append(np.zeros((64, 64, 3)))

        # Imagen generada
        torch.manual_seed(42)
        c = torch.tensor(cond_vecs[idx], dtype=torch.float32).unsqueeze(0).to(device)
        m = torch.tensor(masks_arr[idx], dtype=torch.float32).unsqueeze(0).to(device)
        v = torch.tensor(vp, dtype=torch.float32).unsqueeze(0).to(device)
        z = torch.randn(1, cfg.model.z_dim, device=device)
        with torch.no_grad():
            img = gen(z, c, m, v)
        img_np = (img.squeeze().cpu().numpy() + 1) / 2
        raw_gen = np.transpose(img_np, (1, 2, 0)).clip(0, 1)
        gen_imgs.append(to_fire_composite(raw_gen, vp))
        titles.append(f"Viento {wind_dir}\n{name}")

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, N, figsize=(N * 2.1, 5.2))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Real (CA simulador, t=80) vs. Generado (cGAN Run 48) — 8 direcciones de viento",
        fontsize=12, fontweight="bold", color=TXT, y=1.01,
    )

    for i in range(N):
        # Fila 0 — imagen real
        ax = axes[0, i]
        ax.imshow(real_imgs[i])
        ax.axis("off")
        ax.set_title(titles[i], fontsize=8, color=TXT)
        if i == 0:
            ax.set_ylabel("Real (CA)", fontsize=10, fontweight="bold", color="#555")

        # Fila 1 — imagen generada
        ax = axes[1, i]
        ax.imshow(gen_imgs[i])
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Generado\n(cGAN)", fontsize=10, fontweight="bold", color="#555")

    # Etiquetas de fila
    for row, label in zip([0, 1], ["Real (CA)", "Generado (cGAN)"]):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold", color="#333")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
