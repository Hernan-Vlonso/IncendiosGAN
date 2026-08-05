#!/usr/bin/env python3
"""Figura de progresión temporal del simulador CA.

Busca un incendio con propagación direccional clara (viento ~S o ~N)
y muestra los 4 horizontes t=20,40,60,80 con extracción de fuego
e inferno colormap sobre fondo oscuro.

Salida: docs/figures/progresion_temporal_ca.png

Uso:
    python scripts/fig_progresion_ca.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from src.config import get_config

OUT = Path("docs/figures/progresion_temporal_ca.png")
BG  = "#111122"
TXT = "#EEEEEE"

HORIZONS  = [0, 1, 2, 3]
LABELS    = ["t = 20", "t = 40", "t = 60", "t = 80"]


def extract_fire(img_rgb: np.ndarray) -> np.ndarray:
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    return np.clip((r - g*0.5 - b*0.2)*2.5 + 0.15, 0, 1)


def find_directional_record(ca_root: Path, min_fire_mean=0.20, max_fire_mean=0.55):
    """Busca un record en train cuyos 4 horizontes existan y tenga
    cobertura de fuego moderada en t=80 (patrón direccional visible)."""
    seen = set()
    for f in sorted((ca_root / "train").glob("rec*_var0_h3.png")):
        idx = int(f.stem[3:8])
        if idx in seen:
            continue
        seen.add(idx)
        files = sorted((ca_root / "train").glob(f"rec{idx:05d}_var0_h*.png"))
        if len(files) < 4:
            continue
        img = np.array(Image.open(files[-1]).convert("RGB")) / 255.0
        fm  = extract_fire(img).mean()
        if min_fire_mean <= fm <= max_fire_mean:
            return idx, files
    return None, []


def main():
    cfg = get_config()
    ca  = cfg.paths.ca_images

    idx, files = find_directional_record(ca)
    if not files:
        # Fallback: rec00000
        files = sorted((ca / "train").glob("rec00000_var0_h*.png"))
        idx   = 0
    print(f"Usando rec{idx:05d} — {len(files)} horizontes")

    fire_imgs = []
    for f in sorted(files)[:4]:
        img = np.array(Image.open(f).convert("RGB")) / 255.0
        fire_imgs.append(extract_fire(img))

    # Norma compartida para los 4 frames (misma escala temporal)
    vmax = max(fi.max() for fi in fire_imgs) * 0.95
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 0.15))

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2),
                             gridspec_kw={"wspace": 0.05})
    fig.patch.set_facecolor(BG)
    fig.suptitle("Evolución temporal del CA (un incendio, realización individual)",
                 fontsize=11, fontweight="bold", color=TXT, y=1.02)

    for ax, fire, label in zip(axes, fire_imgs, LABELS):
        ax.imshow(fire, cmap="inferno", norm=norm,
                  origin="upper", interpolation="bilinear")
        ax.set_facecolor(BG)
        ax.set_title(label, fontsize=11, fontweight="bold", color=TXT, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
