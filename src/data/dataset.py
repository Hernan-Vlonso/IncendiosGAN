"""PyTorch Dataset para imágenes del autómata celular con condiciones."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.augmentation import (get_image_transforms, ConditionMaskingTransform,
                                    _ROT_SHIFT, _DIR_FIELD_STARTS)

# Columnas one-hot de Dir_viento en el condition_vector (posiciones 14-21 = 8 dirs físicas)
_DIR_VIENTO_START = 14
_DIR_VIENTO_DIRS  = 8   # N, NE, E, SE, S, SW, W, NW (sin Calma ni Missing)


class FireDataset(Dataset):
    """Dataset de imágenes de propagación con vectores de condición.

    Cada sample contiene:
    - image: tensor (3, 64, 64) normalizado a [-1, 1]
    - condition: tensor (32,) vector de condición (posiblemente enmascarado)
    - mask: tensor (32,) máscara de missingness
    - veg_profile: tensor (10,) perfil de vegetación

    Si el index.json contiene el campo "rotation" (0/1/2/3), aplica
    torch.rot90 determinista a la imagen y remapea Exposicion + Dir_viento
    en el condition vector. La rotación es fija por sample (offline).
    """

    def __init__(self, image_dir: str, condition_vectors: np.ndarray,
                 vegetation_profiles: np.ndarray, masks: np.ndarray,
                 img_size: int = 64, is_train: bool = True,
                 mask_prob: float = 0.3):
        self.image_dir = Path(image_dir)
        self.condition_vectors = condition_vectors
        self.vegetation_profiles = vegetation_profiles
        self.masks = masks
        self.is_train = is_train

        # Cargar índice de imágenes
        self.image_paths = []
        self.record_indices = []
        self.rotations = []      # k de torch.rot90 por sample (0 = sin rotación)
        self._load_image_index()

        # Transforms
        self.img_transform = get_image_transforms(img_size, is_train)
        self.cond_masking = ConditionMaskingTransform(mask_prob) if is_train else None

    def _load_image_index(self):
        """Cargar índice de imágenes y sus registros asociados."""
        index_file = self.image_dir / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
            for entry in index:
                self.image_paths.append(self.image_dir / entry["filename"])
                self.record_indices.append(entry["record_idx"])
                self.rotations.append(entry.get("rotation", 0))
        else:
            # Fallback: escanear directorio
            for img_path in sorted(self.image_dir.glob("*.png")):
                self.image_paths.append(img_path)
                parts = img_path.stem.split("_")
                rec_idx = int(parts[0].replace("rec", ""))
                self.record_indices.append(rec_idx)
                self.rotations.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Cargar imagen
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.img_transform(img)

        # Obtener condición y máscara del registro asociado
        rec_idx = self.record_indices[idx]
        condition = self.condition_vectors[rec_idx].copy()
        mask = self.masks[rec_idx].copy()
        veg_profile = self.vegetation_profiles[rec_idx].copy()

        # Rotación determinista offline (campo "rotation" del index.json)
        k = self.rotations[idx]
        if k != 0:
            img = torch.rot90(img, k, dims=[1, 2])
            shift = _ROT_SHIFT[k]
            for start in _DIR_FIELD_STARTS:
                block = condition[start:start + 8].copy()
                condition[start:start + 8] = np.roll(block, shift)

        # Aplicar condition masking durante entrenamiento
        if self.cond_masking is not None:
            condition, mask = self.cond_masking(condition, mask)

        return {
            "image": img,
            "condition": torch.from_numpy(condition),
            "mask": torch.from_numpy(mask),
            "veg_profile": torch.from_numpy(veg_profile),
        }

    def get_effective_dir_indices(self) -> np.ndarray:
        """Devuelve el índice de dirección efectivo (post-rotación) por sample.

        Necesario para calcular el WeightedRandomSampler sobre la dirección
        real de cada sample, no sobre la dirección original del registro.

        Índices 0-7: direcciones físicas (N, NE, E, SE, S, SW, W, NW).
        Índices 8-9: Calma y Missing — se devuelven sin rotar y el sampler
        les asigna peso 0.2× (ver _build_wind_sampler).
        """
        dirs = []
        for i, rec_idx in enumerate(self.record_indices):
            # Solo las 8 direcciones físicas para determinar si rotar
            onehot_physical = self.condition_vectors[
                rec_idx, _DIR_VIENTO_START:_DIR_VIENTO_START + _DIR_VIENTO_DIRS
            ]
            onehot_full = self.condition_vectors[
                rec_idx, _DIR_VIENTO_START:_DIR_VIENTO_START + 10
            ]
            k = self.rotations[i]
            if onehot_physical.sum() > 0.5:
                # Dirección física conocida: rotar si aplica
                orig = int(np.argmax(onehot_physical))
                if k != 0:
                    dirs.append((orig + _ROT_SHIFT[k]) % _DIR_VIENTO_DIRS)
                else:
                    dirs.append(orig)
            else:
                # Calma (8) o Missing (9): no se rota, peso reducido en sampler
                dirs.append(int(np.argmax(onehot_full)))
        return np.array(dirs)


def _build_wind_sampler(dataset: FireDataset) -> WeightedRandomSampler:
    """WeightedRandomSampler que balancea las 8 direcciones físicas de viento.

    Usa la dirección efectiva post-rotación de cada sample, para que el
    sampler sea coherente con el índice expandido (Run 30+).

    Registros con Dir_viento=Missing o Calma reciben peso 0.2× para no
    dominar el batch pero tampoco ser eliminados.
    """
    wind_classes = dataset.get_effective_dir_indices()

    # Contar imágenes por clase
    class_counts = np.zeros(10, dtype=np.float64)
    for c in range(10):
        class_counts[c] = max((wind_classes == c).sum(), 1)

    # Peso por clase
    class_weights = np.zeros(10, dtype=np.float64)
    for c in range(_DIR_VIENTO_DIRS):      # 0-7: dirs físicas
        class_weights[c] = 1.0 / class_counts[c]
    for c in range(_DIR_VIENTO_DIRS, 10):  # 8-9: Calma, Missing
        class_weights[c] = 0.2 / class_counts[c]

    sample_weights = torch.tensor(
        [class_weights[c] for c in wind_classes], dtype=torch.float64
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(cfg, condition_vectors, vegetation_profiles, masks):
    """Crear DataLoaders de train y val."""
    train_dataset = FireDataset(
        image_dir=str(cfg.paths.ca_images / "train"),
        condition_vectors=condition_vectors,
        vegetation_profiles=vegetation_profiles,
        masks=masks,
        img_size=cfg.model.img_size,
        is_train=True,
        mask_prob=cfg.training.mask_prob,
    )

    val_dataset = FireDataset(
        image_dir=str(cfg.paths.ca_images / "val"),
        condition_vectors=condition_vectors,
        vegetation_profiles=vegetation_profiles,
        masks=masks,
        img_size=cfg.model.img_size,
        is_train=False,
        mask_prob=0.0,
    )

    if cfg.training.use_weighted_sampler:
        # WRS sobre dirección efectiva post-rotación
        sampler = _build_wind_sampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} imágenes, Val: {len(val_dataset)} imágenes")
    return train_loader, val_loader
