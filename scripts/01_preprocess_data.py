#!/usr/bin/env python3
"""Script 01: Preprocesamiento del dataset de incendios.

Carga Incendios_UTM.xlsx, limpia, imputa, codifica y normaliza.
Genera: condition_vectors.npy, vegetation_profiles.npy, masks.npy, scaler.pkl
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data.preprocessing import preprocess


def main():
    cfg = get_config()

    print("=" * 60)
    print("PREPROCESAMIENTO DE DATOS")
    print("=" * 60)

    result = preprocess(cfg)

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Registros: {result['metadata']['n_records']}")
    print(f"Condition vector dim: {result['condition_vectors'].shape[1]}")
    print(f"Vegetation profile dim: {result['vegetation_profiles'].shape[1]}")
    print(f"Mask dim: {result['masks'].shape[1]}")
    print(f"\nTasas de missingness por variable continua:")
    for col, rate in zip(result['metadata']['continuous_cols'],
                         result['metadata']['miss_rate']):
        print(f"  {col}: {rate:.1%}")

    print(f"\nArtefactos guardados en: {cfg.paths.processed_data}")


if __name__ == "__main__":
    main()
