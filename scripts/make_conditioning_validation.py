"""
Figura de validación de condicionamiento direccional — Run 48.
Muestra 3 realizaciones para un incendio real por cada dirección de viento
(N, NE, E, SE, S, SW, W, NW), con flecha indicando la dirección de propagación
esperada (downwind = opuesto al origen del viento).

Salida: outputs/figures/fig_conditioning_validation.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import get_config
from src.models.generator import Generator

CHECKPOINT = Path("outputs/checkpoints/run48/checkpoint_best_final.pt")
OUT = Path("outputs/figures/fig_conditioning_validation.png")

# Un incendio representativo por dirección (idx en incendios_clean.csv)
CASES = [
    ("N",  5247, "BATUCO / STA ISABEL"),
    ("NE", 2915, "TABUNCO / EL AGUILA"),
    ("E",  2939, "SANTA CRUZ"),
    ("SE", 2956, "CORONEL DE MAULE"),
    ("S",  2940, "LAS MAQUINAS"),
    ("SW", 5249, "QUIVOLGO 1"),
    ("W",  2992, "LAS CARDILLAS"),
    ("NW", 4932, "CORRAL DE PEREZ"),
]

N_SAMPLES = 3

# Dirección de propagación esperada: downwind (opuesto al origen del viento)
# En coordenadas de imagen: fila 0 = arriba = Norte
# dx = columna (+ = derecha = E), dy = fila (- = arriba = N)
PROPAGATION = {
    "N":  (0,  1),   # viento del N → fuego va al S  (abajo)
    "NE": (-1, 1),   # viento del NE → fuego va al SW
    "E":  (-1, 0),   # viento del E  → fuego va al W  (izquierda)
    "SE": (-1, -1),  # viento del SE → fuego va al NW
    "S":  (0, -1),   # viento del S  → fuego va al N  (arriba)
    "SW": (1, -1),   # viento del SW → fuego va al NE
    "W":  (1,  0),   # viento del W  → fuego va al E  (derecha)
    "NW": (1,  1),   # viento del NW → fuego va al SE
}

COMPASS = {
    "N":  "→S",  "NE": "→SW", "E":  "→W",  "SE": "→NW",
    "S":  "→N",  "SW": "→NE", "W":  "→E",  "NW": "→SE",
}


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cond_vecs = np.load(cfg.paths.condition_vectors)
    veg_profs  = np.load(cfg.paths.vegetation_profiles)
    masks_arr  = np.load(cfg.paths.masks)

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
    print(f"Run 48 epoch {ckpt.get('epoch', '?')} cargado.")

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
                imgs.append(np.transpose(img, (1, 2, 0)))
        return imgs

    nrows = len(CASES)
    ncols = N_SAMPLES + 1   # +1 para etiqueta lateral
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * N_SAMPLES + 2.5, 2.8 * nrows),
                             gridspec_kw={"width_ratios": [1.5] + [1] * N_SAMPLES})

    for row, (wind_dir, idx, fire_name) in enumerate(CASES):
        import pandas as pd
        df = pd.read_csv(cfg.paths.processed_csv)
        rec = df.iloc[idx]
        sup = rec.get("Sup_Total", 0)

        imgs = generate(idx, seed=42)
        dx, dy = PROPAGATION[wind_dir]  # dirección de propagación en imagen

        # Columna 0: etiqueta
        ax_lbl = axes[row][0]
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.65,
                    f"Viento: {wind_dir}\n({COMPASS[wind_dir]})",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    transform=ax_lbl.transAxes)
        ax_lbl.text(0.5, 0.28,
                    f"{fire_name}\n{sup:.0f} ha",
                    ha="center", va="center", fontsize=8, color="#444",
                    transform=ax_lbl.transAxes)

        for col in range(N_SAMPLES):
            ax = axes[row][col + 1]
            ax.imshow(imgs[col], origin="upper", interpolation="nearest")
            ax.axis("off")

            if row == 0:
                ax.set_title(f"Realiz. {col+1}", fontsize=9)

            # Flecha de dirección de propagación esperada
            H, W = imgs[col].shape[:2]
            cx, cy = W / 2, H / 2
            scale = H * 0.28
            ax.annotate("",
                        xy=(cx + dx * scale, cy + dy * scale),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="-|>", color="cyan",
                                        lw=2, mutation_scale=13))

    plt.suptitle(
        "Validación de condicionamiento — Run 48 (época 70)\n"
        "Un incendio real por dirección de viento — flecha cian: propagación esperada (downwind)",
        fontsize=12, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
