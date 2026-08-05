"""
Script 05: Expande el índice del train set con rotaciones deterministas 0/90/180/270°.

Para cada entrada en train/index.json agrega 3 entradas con rotation=1/2/3.
El val set NO se modifica (se evalúa siempre sobre imágenes originales).

Balance resultante por dirección física:
  Cardinales   N/E/S/W   → suma de los 4 cardinales  originales
  Intercardinals NE/SE/SW/NW → suma de los 4 intercardinals originales

Con la distribución del dataset actual esto produce ~336 samples/cardinal
y ~184 samples/intercardinal, vs el rango 11–190 original.

El índice ampliado se guarda como index.json (backup en index_original.json).
Ejecutar UNA sola vez por dataset; es idempotente si ya tiene campo rotation.

Uso:
    python scripts/05_expand_dataset_rotations.py
    python scripts/05_expand_dataset_rotations.py --dry-run
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.config import get_config
from src.data.preprocessing import DIR_VIENTO_CATS

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_DIR_VIENTO_START = 14
_ROT_SHIFT = {0: 0, 1: 2, 2: 4, 3: 6}


def effective_dir(condition_vectors, rec_idx: int, rotation: int) -> int:
    """Índice de dirección efectivo tras aplicar la rotación."""
    onehot = condition_vectors[rec_idx, _DIR_VIENTO_START:_DIR_VIENTO_START + 10]
    orig = int(np.argmax(onehot))
    if orig >= 8:          # Calma o Missing → no rota
        return orig
    return (orig + _ROT_SHIFT[rotation]) % 8


def expand_index(index_path: Path, condition_vectors: np.ndarray,
                 dry_run: bool = False):
    """Lee index.json y escribe versión expandida 4×."""
    with open(index_path) as f:
        original = json.load(f)

    # Detectar si ya está expandido
    if original and "rotation" in original[0]:
        print(f"  {index_path}: ya tiene campo 'rotation', no se modifica.")
        return

    expanded = []
    for entry in original:
        for k in range(4):
            new_entry = dict(entry)
            new_entry["rotation"] = k
            expanded.append(new_entry)

    print(f"  {index_path.parent.name}/index.json: "
          f"{len(original)} -> {len(expanded)} entradas")

    # Estadísticas por dirección efectiva
    dir_counts = Counter()
    for entry in expanded:
        d = effective_dir(condition_vectors, entry["record_idx"], entry["rotation"])
        if d < 8:
            dir_counts[DIRECTIONS[d]] += 1

    print("  Distribución por dirección:")
    for d in DIRECTIONS:
        print(f"    {d:3s}: {dir_counts[d]}")

    if dry_run:
        print("  [dry-run] No se escribe nada.")
        return

    # Backup del índice original
    backup = index_path.parent / "index_original.json"
    if not backup.exists():
        shutil.copy(index_path, backup)
        print(f"  Backup guardado: {backup.name}")

    with open(index_path, "w") as f:
        json.dump(expanded, f)
    print(f"  index.json actualizado.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Solo muestra estadísticas, no escribe.")
    args = p.parse_args()

    cfg = get_config()
    condition_vectors = np.load(cfg.paths.condition_vectors)

    train_index = cfg.paths.ca_images / "train" / "index.json"
    print("Expandiendo train index...")
    expand_index(train_index, condition_vectors, dry_run=args.dry_run)

    print("\nVal index: sin cambios (evaluación sobre imágenes originales).")


if __name__ == "__main__":
    main()
