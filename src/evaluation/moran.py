"""Índice de Moran para evaluación de autocorrelación espacial.

Compara el patrón espacial de las imágenes reales del CA vs las generadas.
Un índice de Moran similar entre real y generado indica que el modelo
reproduce correctamente la estructura espacial de propagación.

Índice de Moran I:
    I = 1 cuando los valores altos están agrupados (autocorrelación positiva)
    I ≈ 0 cuando el patrón es aleatorio
    I = -1 cuando los valores altos están dispersos (autocorrelación negativa)

Para imágenes de propagación esperamos I > 0 (el fuego se propaga en zonas
contiguas, no de forma aleatoria pixel a pixel).
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch

import libpysal
from esda.moran import Moran


def image_to_intensity(img_tensor: torch.Tensor) -> np.ndarray:
    """Convierte tensor de imagen [-1,1] o [0,1] a mapa de intensidad 2D.

    Usa el canal de luminancia (promedio RGB) como proxy de probabilidad de quema.
    Píxeles más brillantes = mayor probabilidad de quema.
    """
    img = img_tensor.cpu().float()
    if img.min() < 0:
        img = (img + 1) / 2  # [-1,1] -> [0,1]
    # Promedio de canales RGB -> luminancia
    if img.dim() == 3:
        intensity = img.mean(dim=0).numpy()  # (H, W)
    else:
        intensity = img.numpy()
    return intensity.astype(np.float64)


def compute_moran(intensity: np.ndarray) -> tuple[float, float]:
    """Calcula el índice de Moran I para una imagen de intensidad 2D.

    Args:
        intensity: array (H, W) con valores en [0, 1]

    Returns:
        (moran_I, p_value)
    """
    h, w = intensity.shape
    flat = intensity.ravel()

    # Matriz de pesos espaciales: vecindad de reina (8 vecinos)
    w_queen = libpysal.weights.lat2W(h, w, rook=False)
    w_queen.transform = "r"  # row-standardize

    mi = Moran(flat, w_queen)
    return float(mi.I), float(mi.p_sim)


def compute_moran_batch(images: list[np.ndarray]) -> dict:
    """Calcula Moran I para un batch de imágenes.

    Args:
        images: lista de arrays (H, W) con intensidades [0, 1]

    Returns:
        dict con mean, std, median, p_value_mean de Moran I
    """
    morans, pvals = [], []
    for img in images:
        I, p = compute_moran(img)
        morans.append(I)
        pvals.append(p)

    return {
        "moran_mean": float(np.mean(morans)),
        "moran_std":  float(np.std(morans)),
        "moran_median": float(np.median(morans)),
        "p_value_mean": float(np.mean(pvals)),
        "n_images": len(morans),
        "values": morans,
    }


def load_ca_images(ca_dir: str, n_samples: int = 200, seed: int = 42) -> list[np.ndarray]:
    """Carga imágenes del CA real y las convierte a intensidad.

    Args:
        ca_dir: directorio con imágenes PNG del CA
        n_samples: número de imágenes a muestrear
        seed: semilla para reproducibilidad

    Returns:
        lista de arrays (H, W) con intensidades [0, 1]
    """
    rng = np.random.default_rng(seed)
    paths = sorted(Path(ca_dir).glob("*.png"))
    if len(paths) > n_samples:
        paths = rng.choice(paths, size=n_samples, replace=False).tolist()

    images = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        intensity = img.mean(axis=2)  # luminancia
        images.append(intensity)
    return images


def evaluate_moran(
    real_images: list[np.ndarray],
    gen_images: list[np.ndarray],
) -> dict:
    """Compara el índice de Moran entre imágenes reales y generadas.

    Args:
        real_images: lista de arrays (H, W) del CA real
        gen_images: lista de arrays (H, W) generadas por la GAN

    Returns:
        dict con resultados para ambos conjuntos y la diferencia
    """
    print(f"Calculando Moran I para {len(real_images)} imágenes reales...")
    real_stats = compute_moran_batch(real_images)

    print(f"Calculando Moran I para {len(gen_images)} imágenes generadas...")
    gen_stats = compute_moran_batch(gen_images)

    diff = abs(real_stats["moran_mean"] - gen_stats["moran_mean"])

    return {
        "real": real_stats,
        "generated": gen_stats,
        "moran_diff": diff,
    }


def print_moran_report(results: dict):
    """Imprime un reporte legible del análisis de Moran."""
    r = results["real"]
    g = results["generated"]

    print("\n" + "=" * 60)
    print("ÍNDICE DE MORAN — Autocorrelación Espacial")
    print("=" * 60)
    print(f"{'':30s} {'Real (CA)':>12s} {'Generado':>12s}")
    print("-" * 60)
    print(f"{'Moran I (media)':30s} {r['moran_mean']:>12.4f} {g['moran_mean']:>12.4f}")
    print(f"{'Moran I (std)':30s} {r['moran_std']:>12.4f} {g['moran_std']:>12.4f}")
    print(f"{'Moran I (mediana)':30s} {r['moran_median']:>12.4f} {g['moran_median']:>12.4f}")
    print(f"{'p-value (media)':30s} {r['p_value_mean']:>12.4f} {g['p_value_mean']:>12.4f}")
    print(f"{'N imágenes':30s} {r['n_images']:>12d} {g['n_images']:>12d}")
    print("-" * 60)
    print(f"{'Diferencia |delta I|':30s} {results['moran_diff']:>12.4f}")
    print("=" * 60)

    print("\nInterpretación:")
    print(f"  Real CA:   I={r['moran_mean']:.4f} -> ", end="")
    _interpret(r["moran_mean"])
    print(f"  Generado:  I={g['moran_mean']:.4f} -> ", end="")
    _interpret(g["moran_mean"])

    if results["moran_diff"] < 0.05:
        print(f"\n  OK Diferencia pequeña (|delta I|={results['moran_diff']:.4f}): el modelo")
        print("    reproduce bien la autocorrelación espacial del CA.")
    elif results["moran_diff"] < 0.15:
        print(f"\n  ~~ Diferencia moderada (|delta I|={results['moran_diff']:.4f}): el modelo")
        print("    aproxima la estructura espacial pero con diferencias.")
    else:
        print(f"\n  XX Diferencia alta (|delta I|={results['moran_diff']:.4f}): el modelo")
        print("    no reproduce bien la autocorrelación espacial del CA.")


def _interpret(I: float):
    if I > 0.5:
        print("alta autocorrelación positiva (fuego muy agrupado)")
    elif I > 0.2:
        print("autocorrelación positiva moderada")
    elif I > 0.05:
        print("autocorrelación positiva leve")
    elif I > -0.05:
        print("patrón aleatorio")
    else:
        print("autocorrelación negativa (patrón disperso)")
