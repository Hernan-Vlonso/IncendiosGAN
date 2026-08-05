#!/usr/bin/env python3
"""Script 02: Generar dataset de imágenes con el autómata celular.

Lee datos preprocesados y genera ~98K imágenes de propagación de fuego.
Uso: python scripts/02_generate_ca_dataset.py [--max-records N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.ca.generate_dataset import generate_full_dataset


def main():
    parser = argparse.ArgumentParser(description="Generar dataset de imágenes CA")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Límite de registros (para pruebas)")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Usar snapshots discretos en vez de heatmaps Monte Carlo")
    parser.add_argument("--mc-runs", type=int, default=None,
                        help="Corridas Monte Carlo por variación (default: 30)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Procesos paralelos (default: CPU cores - 1)")
    args = parser.parse_args()

    cfg = get_config()
    if args.no_heatmap:
        cfg.ca.heatmap_mode = False
    if args.mc_runs:
        cfg.ca.n_monte_carlo_runs = args.mc_runs

    print("=" * 60)
    print("GENERACIÓN DE DATASET DE IMÁGENES CA")
    print("=" * 60)

    train_entries, val_entries = generate_full_dataset(
        cfg, max_records=args.max_records, n_workers=args.workers
    )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Imágenes de entrenamiento: {len(train_entries)}")
    print(f"Imágenes de validación: {len(val_entries)}")
    print(f"Total: {len(train_entries) + len(val_entries)}")
    print(f"Guardadas en: {cfg.paths.ca_images}")


if __name__ == "__main__":
    main()
