"""Preprocesamiento del dataset de incendios.

Carga, limpieza, imputación, encoding y normalización.
Produce: condition vectors (N, 32), vegetation profiles (N, 10), masks (N, 32).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Config, get_config


# Mapeo columnas raw → nombres internos
_COL_RENAME = {
    "Temperatura °C": "Temperatura",
    "Humedad %": "Humedad_Relativa",
    "Velocidad viento Km/hra": "Vel_Viento",
    "Superficie total Has": "Sup_Total",
    "Dirección viento": "Dir_viento",
    "Exposición": "Exposicion",
    "Topografía": "Topografia",
}

# Mapeo columnas raw de vegetación → nombres internos
_VEG_RENAME = {
    "Arbolado": "Arbolado_Nat",
    "Eucalipto": "Arbolado_Exot",
    "Otras plantaciones": "Arbolado_Mixto",
    "Matorral": "Matorral",
    "Pastizal": "Pastizal",
    "Agricola": "Agricola",
    "Desechos": "Desechos",
    "Pino 0 a 10": "Plantacion_Joven",
}

# Categorías esperadas
EXPOSICION_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Plano", "Missing"]
DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
TOPOGRAFIA_CATS = ["Suave", "Irregular", "Abrupta", "Missing"]


def load_raw_data(cfg: Config) -> pd.DataFrame:
    """Cargar datos crudos desde Excel."""
    df = pd.read_excel(cfg.paths.excel_file, engine="openpyxl")
    print(f"Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombrar columnas a nombres internos."""
    df = df.rename(columns=_COL_RENAME)
    return df


def create_missingness_mask(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Crear máscara binaria de missingness para las columnas de condición.

    mask[i, j] = 1 si el dato original estaba ausente.
    La máscara tiene la misma dimensión que el condition vector (32).
    """
    cont_cols = cfg.data.continuous_cols
    cat_cols = cfg.data.categorical_cols
    n = len(df)

    # 4 flags para continuos
    cont_mask = np.zeros((n, len(cont_cols)), dtype=np.float32)
    for i, col in enumerate(cont_cols):
        cont_mask[:, i] = df[col].isna().values.astype(np.float32)

    # Expandir flags categóricos al espacio one-hot
    cat_masks = []
    cat_sizes = {"Exposicion": len(EXPOSICION_CATS), "Dir_viento": len(DIR_VIENTO_CATS),
                 "Topografia": len(TOPOGRAFIA_CATS)}
    for col in cat_cols:
        n_cats = cat_sizes[col]
        is_missing = df[col].isna().values.astype(np.float32)
        # Repetir el flag para cada dimensión one-hot
        cat_masks.append(np.tile(is_missing[:, None], (1, n_cats)))

    # 4 flags adicionales de missingness (1 por continuo)
    miss_flags = cont_mask.copy()

    mask = np.concatenate([cont_mask] + cat_masks + [miss_flags], axis=1)
    return mask


def impute_continuous(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Imputar columnas continuas con mediana."""
    for col in cfg.data.continuous_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def impute_categorical(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Imputar columnas categóricas con 'Missing'."""
    for col in cfg.data.categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")
    return df


def encode_categorical(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """One-hot encoding de columnas categóricas.

    Returns array (N, 24): Exposicion(10) + Dir_viento(10) + Topografia(4)
    """
    cat_arrays = []
    cat_info = [
        ("Exposicion", EXPOSICION_CATS),
        ("Dir_viento", DIR_VIENTO_CATS),
        ("Topografia", TOPOGRAFIA_CATS),
    ]

    for col, categories in cat_info:
        values = df[col].values
        onehot = np.zeros((len(df), len(categories)), dtype=np.float32)
        for i, cat in enumerate(categories):
            onehot[:, i] = (values == cat).astype(np.float32)
        # Si algún valor no matchea ninguna categoría, asignar a "Missing"
        unmatched = onehot.sum(axis=1) == 0
        if unmatched.any():
            miss_idx = categories.index("Missing")
            onehot[unmatched, miss_idx] = 1.0
        cat_arrays.append(onehot)

    return np.concatenate(cat_arrays, axis=1)


def normalize_continuous(df: pd.DataFrame, cfg: Config):
    """Normalizar columnas continuas con StandardScaler.

    Superficie usa log1p antes de escalar.
    Returns: array (N, 4), scaler ajustado.
    """
    cont_cols = cfg.data.continuous_cols
    data = df[cont_cols].values.copy().astype(np.float32)

    # log1p para Sup_Total (última columna)
    sup_idx = cont_cols.index("Sup_Total")
    data[:, sup_idx] = np.log1p(data[:, sup_idx])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data).astype(np.float32)

    return data_scaled, scaler


def build_condition_vector(cont_scaled: np.ndarray, cat_encoded: np.ndarray,
                           miss_flags: np.ndarray) -> np.ndarray:
    """Construir vector de condición (N, 32).

    Layout: [4 continuos | 24 one-hot | 4 missingness flags] = 32 dims
    """
    return np.concatenate([cont_scaled, cat_encoded, miss_flags], axis=1)


def build_vegetation_profile(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Construir perfiles de vegetación normalizados (N, 10).

    Mapea columnas raw a 10 tipos, luego normaliza para que sumen a 1.
    """
    n = len(df)
    profiles = np.zeros((n, 10), dtype=np.float32)

    # Mapear columnas disponibles
    raw_to_idx = {
        "Arbolado": 0,      # Arbolado_Nat
        "Eucalipto": 1,     # Arbolado_Exot
        "Otras plantaciones": 2,  # Arbolado_Mixto
        "Matorral": 3,
        "Pastizal": 4,
        "Agricola": 5,
        "Desechos": 6,
        "Pino 0 a 10": 7,  # Plantacion_Joven
    }

    for raw_col, idx in raw_to_idx.items():
        if raw_col in df.columns:
            profiles[:, idx] = df[raw_col].fillna(0).values.astype(np.float32)

    # Sin_Vegetacion (idx=8): calculada como diferencia
    sup_total = df["Sup_Total"].fillna(0).values.astype(np.float32)
    used = profiles.sum(axis=1)
    # Pino maduro: Pino 11 a 17 + Pino 18 o mas → Otro_Comb (idx=9)
    pino_maduro = (
        df.get("Pino 11 a 17", pd.Series(0, index=df.index)).fillna(0).values +
        df.get("Pino 18 o mas", pd.Series(0, index=df.index)).fillna(0).values
    ).astype(np.float32)
    profiles[:, 9] = pino_maduro
    used += pino_maduro

    # Sin_Vegetacion: superficie sin cobertura
    profiles[:, 8] = np.maximum(0, sup_total - used)

    # Normalizar para que sumen a 1 (proporciones fraccionarias)
    row_sums = profiles.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)  # evitar div/0
    profiles = profiles / row_sums

    return profiles


def preprocess(cfg: Config = None):
    """Pipeline completo de preprocesamiento.

    Returns: dict con condition_vectors, vegetation_profiles, masks, metadata.
    """
    if cfg is None:
        cfg = get_config()

    # 1. Cargar datos
    df = load_raw_data(cfg)

    # 2. Renombrar columnas
    df = rename_columns(df)

    # 3. Crear máscara ANTES de imputar
    cont_cols = cfg.data.continuous_cols
    miss_flags = np.zeros((len(df), len(cont_cols)), dtype=np.float32)
    for i, col in enumerate(cont_cols):
        if col in df.columns:
            miss_flags[:, i] = df[col].isna().values.astype(np.float32)

    mask = create_missingness_mask(df, cfg)

    # 4. Imputar
    df = impute_continuous(df, cfg)
    df = impute_categorical(df, cfg)

    # 5. Encoding categórico
    cat_encoded = encode_categorical(df, cfg)

    # 6. Normalización continua
    cont_scaled, scaler = normalize_continuous(df, cfg)

    # 7. Construir vector de condición
    condition_vectors = build_condition_vector(cont_scaled, cat_encoded, miss_flags)
    print(f"Condition vectors shape: {condition_vectors.shape}")

    # 8. Perfiles de vegetación
    vegetation_profiles = build_vegetation_profile(df, cfg)
    print(f"Vegetation profiles shape: {vegetation_profiles.shape}")

    # 9. Mask
    print(f"Masks shape: {mask.shape}")

    # 10. Metadata adicional
    metadata = {
        "n_records": len(df),
        "continuous_cols": cfg.data.continuous_cols,
        "categorical_cols": cfg.data.categorical_cols,
        "vegetation_cols": cfg.data.vegetation_cols,
        "condition_dim": condition_vectors.shape[1],
        "miss_rate": miss_flags.mean(axis=0).tolist(),
    }

    # Guardar coordenadas para posibles usos futuros
    coord_cols = ["Lat Decimal", "Lon Decimal"]
    coords = None
    if all(c in df.columns for c in coord_cols):
        coords = df[coord_cols].values.astype(np.float32)
        metadata["has_coords"] = True

    # 11. Guardar artefactos
    cfg.paths.ensure_dirs()
    np.save(cfg.paths.condition_vectors, condition_vectors)
    np.save(cfg.paths.vegetation_profiles, vegetation_profiles)
    np.save(cfg.paths.masks, mask)

    with open(cfg.paths.scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(cfg.paths.metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    # Guardar CSV limpio
    df.to_csv(cfg.paths.processed_csv, index=False)
    print(f"Artefactos guardados en {cfg.paths.processed_data}")

    return {
        "condition_vectors": condition_vectors,
        "vegetation_profiles": vegetation_profiles,
        "masks": mask,
        "metadata": metadata,
        "coords": coords,
        "scaler": scaler,
        "df": df,
    }
