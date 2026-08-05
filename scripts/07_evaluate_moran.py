#!/usr/bin/env python3
"""Script 07: Evaluación de autocorrelación espacial con índice de Moran.

Compara la estructura espacial de las imágenes reales del CA
vs las generadas por la cGAN.

Uso:
    python scripts/07_evaluate_moran.py
    python scripts/07_evaluate_moran.py --checkpoint checkpoint_best_run7.pt --n 200
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import get_config
from src.models.generator import Generator
from src.data.dataset import FireDataset
from src.evaluation.moran import (
    load_ca_images, image_to_intensity,
    evaluate_moran, print_moran_report,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint_best_run7.pt",
                        help="Nombre del checkpoint a usar")
    parser.add_argument("--veg_dim", type=int, default=None,
                        help="Dimensión veg_profile. Default: usa cfg.model.veg_dim")
    parser.add_argument("--n", type=int, default=200,
                        help="Número de imágenes a evaluar")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} (veg_dim={args.veg_dim})")

    # ── Cargar datos ────────────────────────────────────────────
    cond  = np.load(cfg.paths.condition_vectors)
    masks = np.load(cfg.paths.masks)
    veg   = np.load(cfg.paths.vegetation_profiles)

    val_dataset = FireDataset(
        str(cfg.paths.ca_images / "val"), cond, veg, masks,
        is_train=False, mask_prob=0.0,
    )

    # ── Cargar generador ────────────────────────────────────────
    veg_dim = args.veg_dim if args.veg_dim is not None else cfg.model.veg_dim
    ckpt_path = cfg.paths.checkpoints / args.checkpoint
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        veg_dim=veg_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema", ckpt.get("generator", ckpt))
    generator.load_state_dict(state)
    generator.eval()
    print(f"Generador cargado (epoch {ckpt.get('epoch', '?')})")

    # ── Imágenes reales del CA ──────────────────────────────────
    print(f"\nCargando {args.n} imágenes reales del CA (val set)...")
    real_ca = load_ca_images(
        str(cfg.paths.ca_images / "val"),
        n_samples=args.n, seed=args.seed,
    )

    # ── Imágenes generadas ──────────────────────────────────────
    print(f"Generando {args.n} imágenes con la GAN...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n = min(args.n, len(val_dataset))
    idx = np.random.choice(len(val_dataset), size=n, replace=False)

    gen_intensities = []
    batch_size = 32
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        c = torch.stack([val_dataset[i]["condition"] for i in batch_idx]).to(device)
        m = torch.stack([val_dataset[i]["mask"] for i in batch_idx]).to(device)
        z = torch.randn(len(batch_idx), cfg.model.z_dim, device=device)

        if veg_dim > 0:
            v = torch.stack([val_dataset[i]["veg_profile"] for i in batch_idx]).to(device)
        else:
            v = torch.zeros(len(batch_idx), 0, device=device)

        with torch.no_grad():
            imgs = generator(z, c, m, v)

        for img in imgs:
            intensity = image_to_intensity(img)
            gen_intensities.append(intensity)

    # ── Calcular Moran ──────────────────────────────────────────
    results = evaluate_moran(real_ca, gen_intensities)
    print_moran_report(results)

    # ── Guardar resultados ──────────────────────────────────────
    out_path = cfg.paths.metrics / "moran_results.json"
    save_results = {
        "checkpoint": args.checkpoint,
        "n_images": args.n,
        "real": {k: v for k, v in results["real"].items() if k != "values"},
        "generated": {k: v for k, v in results["generated"].items() if k != "values"},
        "moran_diff": results["moran_diff"],
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResultados guardados en: {out_path}")

    # ── Figura: distribución de Moran I ────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(results["real"]["values"], bins=30, alpha=0.6,
            color="tab:green", label=f"Real CA  (μ={results['real']['moran_mean']:.4f})")
    ax.hist(results["generated"]["values"], bins=30, alpha=0.6,
            color="tab:orange", label=f"Generado (μ={results['generated']['moran_mean']:.4f})")
    ax.axvline(results["real"]["moran_mean"], color="tab:green",
               linestyle="--", linewidth=2)
    ax.axvline(results["generated"]["moran_mean"], color="tab:orange",
               linestyle="--", linewidth=2)
    ax.set_xlabel("Índice de Moran I", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Distribución del Índice de Moran — Real vs Generado", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = cfg.paths.figures / "fig_moran_distribution.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figura guardada en: {fig_path}")


if __name__ == "__main__":
    main()
