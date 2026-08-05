#!/usr/bin/env python3
"""Script 03b: Entrenar modelo BASELINE sin condition masking.

Igual que 03_train_gan.py pero con mask_prob=0.0 para el ablation study.
Guarda checkpoints en outputs/checkpoints_baseline/.

Uso: python scripts/03b_train_baseline.py [--epochs N] [--resume PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import get_config
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback, SamplingCallback, MetricsLogger,
)


def main():
    parser = argparse.ArgumentParser(description="Entrenar Baseline (sin masking)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Número de epochs (default: config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a checkpoint para continuar")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Tamaño de batch")
    args = parser.parse_args()

    cfg = get_config()
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    # ── BASELINE: desactivar condition masking ──
    cfg.training.mask_prob = 0.0

    # Directorios separados para baseline
    baseline_ckpt_dir = cfg.paths.project_root / "outputs" / "checkpoints_baseline"
    baseline_samples_dir = cfg.paths.project_root / "outputs" / "samples_baseline"
    baseline_metrics_dir = cfg.paths.project_root / "outputs" / "metrics_baseline"
    baseline_ckpt_dir.mkdir(parents=True, exist_ok=True)
    baseline_samples_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Semilla
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    print("=" * 60)
    print("ENTRENAMIENTO BASELINE (sin condition masking)")
    print(f"mask_prob = {cfg.training.mask_prob}")
    print("=" * 60)

    # Cargar datos preprocesados
    print("\nCargando datos preprocesados...")
    condition_vectors = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)
    masks = np.load(cfg.paths.masks)

    # Crear dataloaders
    train_loader, val_loader = create_dataloaders(
        cfg, condition_vectors, vegetation_profiles, masks
    )

    # Callbacks (rutas de baseline)
    callbacks = [
        CheckpointCallback(
            str(baseline_ckpt_dir),
            interval=cfg.training.checkpoint_interval,
        ),
        SamplingCallback(
            str(baseline_samples_dir),
            n_samples=cfg.training.n_sample_images,
            interval=cfg.training.sample_interval,
        ),
        MetricsLogger(str(baseline_metrics_dir / "training_log.csv")),
    ]

    # Crear trainer
    trainer = Trainer(cfg, train_loader, val_loader, callbacks)

    # Reanudar si se especifica
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Entrenar
    trainer.train()

    print(f"\nCheckpoints en: {baseline_ckpt_dir}")
    print(f"Samples en: {baseline_samples_dir}")
    print(f"Métricas en: {baseline_metrics_dir}")


if __name__ == "__main__":
    main()
