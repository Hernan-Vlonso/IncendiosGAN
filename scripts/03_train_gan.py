#!/usr/bin/env python3
"""Script 03: Entrenar la Conditional DCGAN con WGAN-GP.

Uso: python scripts/03_train_gan.py [--epochs N] [--resume PATH]
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
    BestCheckpointCallback, BestDirectionalCallback,
)


def main():
    parser = argparse.ArgumentParser(description="Entrenar Conditional DCGAN")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Número de epochs (default: config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta a checkpoint para continuar")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Tamaño de batch")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Identificador del run (ej: run29). Aísla checkpoints, "
                             "samples y log para no solapar runs anteriores.")
    parser.add_argument("--save-best-final", type=str, default=None,
                        help="Al terminar el entrenamiento, copia el checkpoint de "
                             "mayor directional alignment a esta ruta "
                             "(ej: checkpoint_best_final.pt). Equivale a la "
                             "selección manual del mejor checkpoint del run.")
    args = parser.parse_args()

    cfg = get_config()
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size

    # Redirigir outputs a subdirectorios por run para evitar solapamiento
    if args.run_id:
        cfg.paths.checkpoints = cfg.paths.checkpoints / args.run_id
        cfg.paths.samples     = cfg.paths.samples     / args.run_id
        cfg.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        cfg.paths.samples.mkdir(parents=True, exist_ok=True)

    # Semilla
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    print("=" * 60)
    print("ENTRENAMIENTO - Conditional DCGAN con Hinge + R1")
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

    # Callbacks
    callbacks = [
        CheckpointCallback(
            str(cfg.paths.checkpoints),
            interval=cfg.training.checkpoint_interval,
        ),
        SamplingCallback(
            str(cfg.paths.samples),
            n_samples=cfg.training.n_sample_images,
            interval=cfg.training.sample_interval,
        ),
        BestCheckpointCallback(
            str(cfg.paths.checkpoints),
            n_samples=32,
            interval=cfg.training.sample_interval,
        ),
        BestDirectionalCallback(
            str(cfg.paths.checkpoints),
            n_per_dir=4,
            interval=cfg.training.sample_interval,
        ),
        MetricsLogger(str(cfg.paths.metrics / f"training_log_{args.run_id}.csv"
                          if args.run_id else cfg.paths.metrics / "training_log.csv")),
    ]

    # Crear trainer
    trainer = Trainer(cfg, train_loader, val_loader, callbacks)

    # Reanudar si se especifica
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Entrenar
    trainer.train()

    # Copiar el mejor checkpoint direccional a ruta nombrada si se indica
    if args.save_best_final:
        import shutil
        best_dir_src = cfg.paths.checkpoints / "checkpoint_best_dir.pt"
        dest = Path(args.save_best_final)
        if best_dir_src.exists():
            shutil.copy2(best_dir_src, dest)
            print(f"\nMejor checkpoint guardado como: {dest}")
        else:
            print(f"\nADVERTENCIA: no se encontró {best_dir_src}; "
                  "checkpoint_best_final.pt no generado.")

    print(f"\nCheckpoints en: {cfg.paths.checkpoints}")
    print(f"Samples en: {cfg.paths.samples}")
    print(f"Métricas en: {cfg.paths.metrics}")


if __name__ == "__main__":
    main()
