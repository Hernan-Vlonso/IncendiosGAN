"""
Genera figura comparativa SAN JOSE: Dir_viento=S (completo) vs Dir_viento=Missing.
Usa imágenes raw 64×64 para mostrar claramente la diferencia direccional.

Salida: outputs/figures/geomaps/comparison_sanjose_complete_vs_incomplete.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from src.config import get_config
from src.models.generator import Generator

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "run48" / "checkpoint_best_final.pt"
OUT = ROOT / "outputs" / "figures" / "geomaps" / "comparison_sanjose_complete_vs_incomplete.png"

DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cargar datos procesados
    cond_vecs = np.load(cfg.paths.condition_vectors)
    veg_profs  = np.load(cfg.paths.vegetation_profiles)
    masks_arr  = np.load(cfg.paths.masks)

    # SAN JOSE está en el índice 4
    fire_idx = 4
    cond_s = cond_vecs[fire_idx].copy().astype(np.float32)
    mask_v = masks_arr[fire_idx].copy().astype(np.float32)
    veg    = veg_profs[fire_idx].astype(np.float32)

    # Variante Missing: reemplazar one-hot Dir_viento por "Missing"
    cond_m = cond_s.copy()
    for i in range(10):          # borrar toda la ranura Dir_viento (dims 14-23)
        cond_m[14 + i] = 0.0
    cond_m[14 + DIR_VIENTO_CATS.index("Missing")] = 1.0

    # Cargar generador Run 48
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

    def generate(cond, n=4, seed=42):
        torch.manual_seed(seed)
        c = torch.tensor(cond, dtype=torch.float32).unsqueeze(0).to(device)
        m = torch.tensor(mask_v, dtype=torch.float32).unsqueeze(0).to(device)
        v = torch.tensor(veg, dtype=torch.float32).unsqueeze(0).to(device)
        imgs = []
        with torch.no_grad():
            for _ in range(n):
                z = torch.randn(1, cfg.model.z_dim, device=device)
                img = gen(z, c, m, v)
                img = (img.squeeze().cpu().numpy() + 1) / 2  # [0,1] CHW
                imgs.append(np.transpose(img, (1, 2, 0)))    # HWC
        return imgs

    imgs_s = generate(cond_s, n=4, seed=42)
    imgs_m = generate(cond_m, n=4, seed=42)

    # Figura: 4 filas × 2 columnas
    fig, axes = plt.subplots(4, 2, figsize=(7, 13))

    for i in range(4):
        for j, (imgs, col_title, is_complete) in enumerate([
            (imgs_s, "Dir_viento = S (conocido)\nPropagación hacia el norte (downwind)", True),
            (imgs_m, "Dir_viento = Missing\n(resto de condiciones idéntico)", False),
        ]):
            ax = axes[i][j]
            ax.imshow(imgs[i], origin="upper", interpolation="nearest")
            ax.axis("off")
            if i == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold")
            # Flecha direccional esperada (solo columna completa)
            if is_complete:
                H, W = imgs[i].shape[:2]
                cx, cy = W / 2, H / 2
                ax.annotate("", xy=(cx, cy - 17), xytext=(cx, cy + 4),
                            arrowprops=dict(arrowstyle="-|>", color="cyan",
                                            lw=2, mutation_scale=14))
                ax.text(cx, cy - 21, "N ↑", color="cyan", ha="center",
                        va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle(
        "Incendio SAN JOSE — 12 ha, Viento Sur 9,8 km/h\n"
        "Efecto de Dir_viento en la propagación generada (Run 48, época 70)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
