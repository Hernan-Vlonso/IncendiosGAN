"""Configuración centralizada del proyecto IncendiosGANs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """Rutas del proyecto."""
    project_root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    processed_data: Path = PROJECT_ROOT / "data" / "processed"
    ca_images: Path = PROJECT_ROOT / "data" / "ca_images"
    checkpoints: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    samples: Path = PROJECT_ROOT / "outputs" / "samples"
    metrics: Path = PROJECT_ROOT / "outputs" / "metrics"
    figures: Path = PROJECT_ROOT / "outputs" / "figures"

    # Archivos específicos
    excel_file: Path = PROJECT_ROOT / "data" / "raw" / "Incendios_UTM.xlsx"
    processed_csv: Path = PROJECT_ROOT / "data" / "processed" / "incendios_clean.csv"
    condition_vectors: Path = PROJECT_ROOT / "data" / "processed" / "condition_vectors.npy"
    vegetation_profiles: Path = PROJECT_ROOT / "data" / "processed" / "vegetation_profiles.npy"
    masks: Path = PROJECT_ROOT / "data" / "processed" / "masks.npy"
    scaler_path: Path = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
    metadata_path: Path = PROJECT_ROOT / "data" / "processed" / "metadata.pkl"

    def ensure_dirs(self):
        """Crear directorios si no existen."""
        for attr in [
            self.processed_data, self.ca_images,
            self.ca_images / "train", self.ca_images / "val",
            self.checkpoints, self.samples, self.metrics, self.figures,
        ]:
            attr.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuración del preprocesamiento de datos."""
    # Columnas continuas del Excel (nombres reales)
    continuous_cols_raw: List[str] = field(default_factory=lambda: [
        "Temperatura °C", "Humedad %", "Velocidad viento Km/hra",
        "Superficie total Has"
    ])
    # Nombres internos normalizados
    continuous_cols: List[str] = field(default_factory=lambda: [
        "Temperatura", "Humedad_Relativa", "Vel_Viento", "Sup_Total"
    ])
    # Columnas categóricas del Excel (nombres reales)
    categorical_cols_raw: List[str] = field(default_factory=lambda: [
        "Exposición", "Dirección viento", "Topografía"
    ])
    # Nombres internos normalizados
    categorical_cols: List[str] = field(default_factory=lambda: [
        "Exposicion", "Dir_viento", "Topografia"
    ])
    # Columnas de vegetación del Excel → perfil de 10 dims
    vegetation_cols_raw: List[str] = field(default_factory=lambda: [
        "Arbolado", "Eucalipto", "Otras plantaciones",
        "Matorral", "Pastizal", "Agricola", "Desechos",
        "Pino 0 a 10", "Pino 11 a 17", "Pino 18 o mas"
    ])
    # Nombres internos del perfil de vegetación
    vegetation_cols: List[str] = field(default_factory=lambda: [
        "Arbolado_Nat", "Arbolado_Exot", "Arbolado_Mixto",
        "Matorral", "Pastizal", "Agricola",
        "Desechos", "Plantacion_Joven", "Sin_Vegetacion", "Otro_Comb"
    ])
    # Dimensionalidad del vector de condición
    condition_dim: int = 32
    # Dimensionalidad del perfil de vegetación
    vegetation_dim: int = 10
    # Fracción para validación
    val_fraction: float = 0.1
    # Semilla para reproducibilidad
    seed: int = 42


@dataclass
class CAConfig:
    """Configuración del autómata celular."""
    grid_size: int = 64
    n_steps: int = 80
    n_snapshots: int = 5  # snapshots por simulación
    n_terrain_variations: int = 3  # variaciones de terreno por registro
    # Parámetros de propagación
    base_spread_prob: float = 0.15
    wind_factor: float = 0.50
    slope_factor: float = 0.15
    humidity_factor: float = 0.4
    temperature_factor: float = 0.2
    # Perlin noise para terreno
    noise_scale: float = 0.08
    noise_octaves: int = 4
    # Punto de ignición
    ignition_radius: int = 2
    # Heatmap Monte Carlo
    heatmap_mode: bool = True  # usar heatmaps en vez de snapshots discretos
    n_monte_carlo_runs: int = 30  # corridas MC por variación de terreno
    n_horizons: int = 4  # horizontes temporales por variación
    min_fire_coverage: float = 0.02
    # Calibración de n_steps por tamaño real del incendio
    n_steps_min: int = 40
    # n_steps ya definido arriba es el máximo (incendios grandes, z~=+2)
    # Flammabilidad por tipo de vegetación (orden: vegetation_cols)
    flammability: List[float] = field(default_factory=lambda: [
        0.6,   # Arbolado_Nat
        0.75,  # Arbolado_Exot
        0.65,  # Arbolado_Mixto
        0.7,   # Matorral
        0.8,   # Pastizal
        0.5,   # Agricola
        0.85,  # Desechos
        0.7,   # Plantacion_Joven
        0.0,   # Sin_Vegetacion
        0.4,   # Otro_Comb
    ])


@dataclass
class ModelConfig:
    """Configuración de la arquitectura del modelo."""
    # Dimensiones
    z_dim: int = 128
    cond_embed_dim: int = 128
    img_channels: int = 3
    img_size: int = 64
    # Generator: 4×4 → 8 → 16 → 32 → 64 (4 bloques de upsampling)
    g_base_channels: int = 512
    g_features: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    # Discriminator: 64 → 32 → 16 → 8 → 4 (4 bloques de downsampling)
    d_base_channels: int = 64
    d_features: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    # Conditioning
    cond_input_dim: int = 64  # condition_dim(32) + mask_dim(32)
    veg_dim: int = 10          # perfil de vegetación (10 tipos), sin máscara
    cond_hidden_dim: int = 256  # Run 13 usó 512 (sin mejora)


@dataclass
class TrainingConfig:
    """Configuración del entrenamiento."""
    # General
    epochs: int = 150
    batch_size: int = 64
    num_workers: int = 4
    # Optimizador
    lr_g: float = 1e-4   # run10: TTUR lr_G < lr_D para convergencia Nash (Heusel 2017)
    lr_d: float = 4e-4   # run10: TTUR lr_D 4x mayor que lr_G
    lr_min: float = 1e-5  # LR mínimo para cosine annealing
    beta1: float = 0.0
    beta2: float = 0.99
    # Hinge loss + R1
    n_critic: int = 4  # Run 42: 4 pasos D por cada paso G para evitar colapso del discriminador
    r1_gamma: float = 10.0  # run20 (modelo definitivo): previene colapso temprano del discriminador
    diversity_weight: float = 0.1   # run13: revertido a óptimo (run12 con 0.15 empeoró ambas métricas)
    r1_interval: int = 1    # cada paso para regularización estable (antes cada 16)
    # Feature matching loss (fuerza a G a reproducir estadísticas de features de D)
    fm_weight: float = 5.0   # 0.0 = desactivado; rango típico 1.0-10.0
    # Gradient clipping para el generador
    g_clip_norm: float = 1.0
    # Condition masking augmentation
    mask_prob: float = 0.3
    # DiffAugment policy (vacío = desactivado)
    diffaug_policy: str = "color,translation,cutout"
    # Auxiliary direction classification loss — AC-GAN (Run 44-48: activado)
    lambda_dir_real: float = 1.0   # CE global en D (Run 48: revertido a global desde per-dir de run47)
    lambda_dir_g: float = 1.0
    # WeightedRandomSampler sobre dirección (Run 30: off, Run 38: on, Run 39: off, Run 40-44: on)
    use_weighted_sampler: bool = True
    # Centroid direction loss (Run 34-43: activo, Run 44: desactivado — reemplazado por AC-GAN)
    lambda_centroid: float = 0.0
    # EMA del generador
    ema_decay: float = 0.999
    # Logging
    log_interval: int = 50  # batches
    sample_interval: int = 5  # epochs
    checkpoint_interval: int = 10  # epochs (reducido para no perder el mejor checkpoint)
    n_sample_images: int = 64
    # Dispositivo
    device: str = "cuda"
    # Semilla
    seed: int = 42


@dataclass
class EvalConfig:
    """Configuración de evaluación."""
    # FID
    fid_n_samples: int = 2400  # protocolo v2: val set completo (2400 imágenes)
    fid_batch_size: int = 32
    # SSIM
    ssim_k_neighbors: int = 5
    ssim_n_samples: int = 500
    # Robustez ante datos incompletos
    masking_percentages: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9
    ])
    robustness_n_samples: int = 600


@dataclass
class Config:
    """Configuración global del proyecto."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ca: CAConfig = field(default_factory=CAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()


def get_config() -> Config:
    """Obtener configuración por defecto."""
    return Config()
