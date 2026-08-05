#!/usr/bin/env python3
"""Genera figura de comparación real vs generado.

Uso:
    python scripts/make_comparison.py
    python scripts/make_comparison.py --checkpoint checkpoint_best_run3.pt --veg_dim 0 --suffix run3
    python scripts/make_comparison.py --checkpoint checkpoint_best_run9.pt --veg_dim 10 --suffix run9
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.config import get_config
from src.models.generator import Generator
from src.data.dataset import FireDataset

N = 8
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint_best_run9.pt")
    parser.add_argument("--veg_dim", type=int, default=10)
    parser.add_argument("--suffix", default="", help="Sufijo para el nombre de archivo (ej: run3)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} (veg_dim={args.veg_dim})")

    # Cargar datos
    cond  = np.load(cfg.paths.condition_vectors)
    masks = np.load(cfg.paths.masks)
    veg   = np.load(cfg.paths.vegetation_profiles)

    val_dataset = FireDataset(
        str(cfg.paths.ca_images / "val"), cond, veg, masks,
        is_train=False, mask_prob=0.0,
    )
    print(f"Val dataset: {len(val_dataset)} imágenes")

    # Cargar generador
    ckpt_path = cfg.paths.checkpoints / args.checkpoint
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        veg_dim=args.veg_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("generator", ckpt))
    generator.load_state_dict(state)
    generator.eval()
    print(f"Checkpoint cargado: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # Seleccionar N muestras variadas
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    indices = np.random.choice(len(val_dataset), size=N, replace=False)

    real_imgs, gen_imgs = [], []
    for idx in indices:
        sample = val_dataset[idx]
        real_imgs.append(sample["image"])

        c = sample["condition"].unsqueeze(0).to(device)
        m = sample["mask"].unsqueeze(0).to(device)
        v = sample["veg_profile"].unsqueeze(0).to(device)
        z = torch.randn(1, cfg.model.z_dim, device=device)

        v_input = sample["veg_profile"].unsqueeze(0).to(device) if args.veg_dim > 0 else torch.zeros(1, 0, device=device)
        with torch.no_grad():
            gen = generator(z, c, m, v_input)
        gen_imgs.append(gen.squeeze(0).cpu())

    def to_np(t):
        img = (t + 1) / 2  # [-1,1] → [0,1]
        return img.permute(1, 2, 0).numpy().clip(0, 1)

    # ── Figura ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, N, figsize=(N * 2.2, 5.5))
    run_label = f" — {args.suffix.upper()}" if args.suffix else ""
    fig.suptitle(f"Real (CA) vs Generado (cGAN){run_label}", fontsize=15, fontweight="bold", y=1.01)

    for i in range(N):
        axes[0, i].imshow(to_np(real_imgs[i]))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Real (CA)", fontsize=11, fontweight="bold")

        axes[1, i].imshow(to_np(gen_imgs[i]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Generado\n(cGAN)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    suffix_str = f"_{args.suffix}" if args.suffix else ""
    out_path = cfg.paths.figures / f"comparacion_real_vs_generado{suffix_str}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nFigura guardada en: {out_path}")

    # ── Segunda figura: progresión temporal de un incendio real ─
    print("\nGenerando progresión temporal...")
    ca_train = list((cfg.paths.ca_images / "train").glob("rec00000_var0_h*.png"))
    ca_train.sort()

    if ca_train:
        fig2, axes2 = plt.subplots(1, len(ca_train), figsize=(len(ca_train) * 2.5, 3))
        horizons = ["t=20", "t=40", "t=60", "t=80"]
        for i, (ax, p) in enumerate(zip(axes2, ca_train)):
            img = Image.open(p)
            ax.imshow(img)
            ax.set_title(horizons[i] if i < len(horizons) else f"h{i}", fontsize=11)
            ax.axis("off")
        fig2.suptitle("Evolución temporal del AC (un incendio)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out2 = cfg.paths.figures / "progresion_temporal_ca.png"
        plt.savefig(str(out2), dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Figura guardada en: {out2}")

if __name__ == "__main__":
    main()
