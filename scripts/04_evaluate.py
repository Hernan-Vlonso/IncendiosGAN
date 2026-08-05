#!/usr/bin/env python3
"""Script 04: Evaluación completa del modelo entrenado.

Calcula FID, SSIM condition-matched, y curvas de robustez.
Uso: python scripts/04_evaluate.py --checkpoint PATH
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import get_config
from src.data.dataset import create_dataloaders
from src.models.generator import Generator
from src.evaluation.evaluator import Evaluator
from src.visualization.plots import plot_robustness_curves


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Ruta al checkpoint del modelo")
    parser.add_argument("--veg_dim", type=int, default=None,
                        help="Override veg_dim (default: usa cfg.model.veg_dim)")
    parser.add_argument("--run_name", type=str, default="",
                        help="Sufijo para el archivo de resultados, e.g. 'run10'")
    parser.add_argument("--fid_n_samples", type=int, default=None,
                        help="Override fid_n_samples (default: usa cfg.eval.fid_n_samples)")
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 60)

    # Cargar datos
    condition_vectors = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)
    masks = np.load(cfg.paths.masks)

    _, val_loader = create_dataloaders(
        cfg, condition_vectors, vegetation_profiles, masks
    )

    # Cargar generador
    print(f"\nCargando checkpoint: {args.checkpoint}")
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        img_channels=cfg.model.img_channels,
        features=cfg.model.g_features,
        veg_dim=args.veg_dim if args.veg_dim is not None else cfg.model.veg_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Usar EMA weights si disponible
    if "ema" in ckpt:
        generator.load_state_dict(ckpt["ema"])
        print("Usando pesos EMA")
    else:
        generator.load_state_dict(ckpt["generator"])

    generator.eval()

    # Evaluar
    if args.veg_dim is not None:
        cfg.model.veg_dim = args.veg_dim
    if args.fid_n_samples is not None:
        cfg.eval.fid_n_samples = args.fid_n_samples
    run_suffix = f"_{args.run_name}" if args.run_name else ""
    evaluator = Evaluator(cfg, device, run_suffix=run_suffix)
    results = evaluator.evaluate_all(
        generator, val_loader, condition_vectors, masks, vegetation_profiles
    )

    # Generar figura de robustez (nombre por run para no sobrescribir)
    rob = results["robustness"]
    plot_robustness_curves(
        rob["masking_levels"], rob["fid_curve"], rob["ssim_curve"],
        save_path=str(cfg.paths.figures / f"robustness_curves{run_suffix}.png"),
    )

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    print(f"FID: {results['fid']:.2f}")
    print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Figuras guardadas en: {cfg.paths.figures}")


if __name__ == "__main__":
    main()
