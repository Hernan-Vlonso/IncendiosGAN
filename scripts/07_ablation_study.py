#!/usr/bin/env python3
"""Script 07: Ablation study — comparar modelo con masking vs baseline.

Evalúa ambos modelos en curvas de robustez y genera figura comparativa.

Uso:
    python scripts/07_ablation_study.py \
        --checkpoint-masking outputs/checkpoints/checkpoint_final.pt \
        --checkpoint-baseline outputs/checkpoints_baseline/checkpoint_final.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import get_config
from src.data.dataset import create_dataloaders
from src.models.generator import Generator
from src.evaluation.fid import FIDCalculator
from src.evaluation.incomplete_data import evaluate_robustness
from src.visualization.plots import plot_ablation_robustness


def load_generator(cfg, checkpoint_path, device):
    """Cargar generador desde checkpoint (usa EMA si disponible)."""
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        img_channels=cfg.model.img_channels,
        features=cfg.model.g_features,
        veg_dim=cfg.model.veg_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema" in ckpt:
        generator.load_state_dict(ckpt["ema"])
        print(f"  Usando pesos EMA (epoch {ckpt.get('epoch', '?')})")
    else:
        generator.load_state_dict(ckpt["generator"])
        print(f"  Usando pesos del generador (epoch {ckpt.get('epoch', '?')})")
    generator.eval()
    return generator


def get_real_images(val_loader, n_samples, device):
    """Extraer imágenes reales del val loader."""
    images = []
    total = 0
    for batch in val_loader:
        imgs = (batch["image"] + 1) / 2  # [-1,1] → [0,1]
        images.append(imgs)
        total += len(imgs)
        if total >= n_samples:
            break
    return torch.cat(images, dim=0)[:n_samples]


def main():
    parser = argparse.ArgumentParser(description="Ablation study: masking vs baseline")
    parser.add_argument("--checkpoint-masking", type=str, required=True,
                        help="Checkpoint del modelo con condition masking")
    parser.add_argument("--checkpoint-baseline", type=str, required=True,
                        help="Checkpoint del modelo baseline (sin masking)")
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("ABLATION STUDY: Condition Masking vs Baseline")
    print(f"Device: {device}")
    print("=" * 60)

    # Cargar datos
    condition_vectors = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)
    masks = np.load(cfg.paths.masks)
    _, val_loader = create_dataloaders(cfg, condition_vectors, vegetation_profiles, masks)

    # Preparar imágenes reales y FID calculator
    fid_calc = FIDCalculator(device, cfg.eval.fid_batch_size)
    real_images = get_real_images(val_loader, cfg.eval.robustness_n_samples, device)
    fid_calc.cache_real_stats(real_images)

    # ── Modelo con masking ──
    print("\n[1/2] Evaluando modelo CON condition masking...")
    gen_masking = load_generator(cfg, args.checkpoint_masking, device)
    rob_masking = evaluate_robustness(
        gen_masking, real_images, condition_vectors, masks,
        cfg, fid_calc, n_samples=cfg.eval.robustness_n_samples, device=device,
    )

    # ── Modelo baseline ──
    print("\n[2/2] Evaluando modelo BASELINE (sin masking)...")
    gen_baseline = load_generator(cfg, args.checkpoint_baseline, device)
    rob_baseline = evaluate_robustness(
        gen_baseline, real_images, condition_vectors, masks,
        cfg, fid_calc, n_samples=cfg.eval.robustness_n_samples, device=device,
    )

    # ── Generar figura comparativa ──
    fig_dir = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(fig_dir / "fig_ablation_robustness.png")

    plot_ablation_robustness(
        masking_levels=rob_masking["masking_levels"],
        fid_masking=rob_masking["fid_curve"],
        ssim_masking=rob_masking["ssim_curve"],
        fid_baseline=rob_baseline["fid_curve"],
        ssim_baseline=rob_baseline["ssim_curve"],
        save_path=save_path,
    )

    # ── Guardar resultados ──
    results = {
        "masking_levels": rob_masking["masking_levels"],
        "masking": {
            "fid_curve": [float(x) if x is not None else None for x in rob_masking["fid_curve"]],
            "ssim_curve": [float(x) for x in rob_masking["ssim_curve"]],
        },
        "baseline": {
            "fid_curve": [float(x) if x is not None else None for x in rob_baseline["fid_curve"]],
            "ssim_curve": [float(x) for x in rob_baseline["ssim_curve"]],
        },
    }
    results_path = cfg.paths.metrics / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Resumen ──
    print("\n" + "=" * 60)
    print("RESULTADOS ABLATION STUDY")
    print("=" * 60)
    print(f"\n{'Masking %':>10} | {'FID (masking)':>14} | {'FID (baseline)':>14} | "
          f"{'SSIM (masking)':>14} | {'SSIM (baseline)':>15}")
    print("-" * 75)
    for i, level in enumerate(rob_masking["masking_levels"]):
        fid_m = rob_masking["fid_curve"][i]
        fid_b = rob_baseline["fid_curve"][i]
        ssim_m = rob_masking["ssim_curve"][i]
        ssim_b = rob_baseline["ssim_curve"][i]
        fid_m_str = f"{fid_m:.2f}" if fid_m is not None else "N/A"
        fid_b_str = f"{fid_b:.2f}" if fid_b is not None else "N/A"
        print(f"{level*100:>9.0f}% | {fid_m_str:>14} | {fid_b_str:>14} | "
              f"{ssim_m:>14.4f} | {ssim_b:>15.4f}")

    print(f"\nFigura guardada en: {save_path}")
    print(f"Resultados guardados en: {results_path}")


if __name__ == "__main__":
    main()
