"""Frechet Inception Distance (FID) para evaluar calidad de imágenes generadas."""

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from tqdm import tqdm


class InceptionFeatureExtractor(nn.Module):
    """Extraer features de InceptionV3 (pool3, 2048-dim)."""

    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        # Hasta el avgpool (antes del clasificador)
        self.model = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Transform para InceptionV3: resize a 299x299, normalizar
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extraer features (batch, 2048).

        Args:
            imgs: tensor (batch, 3, H, W) en rango [0, 1]
        """
        x = self.transform(imgs)
        x = self.model(x)
        return x.squeeze(-1).squeeze(-1)


@torch.no_grad()
def compute_features(images: torch.Tensor, extractor: InceptionFeatureExtractor,
                     batch_size: int = 64) -> np.ndarray:
    """Computar features de InceptionV3 para un conjunto de imágenes.

    Args:
        images: tensor (N, 3, H, W) en rango [0, 1]
        extractor: extractor de features
        batch_size: tamaño de batch

    Returns:
        features: array (N, 2048)
    """
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []

    for (batch,) in tqdm(loader, desc="Extrayendo features", leave=False):
        batch = batch.to(extractor.device)
        feat = extractor(batch)
        features.append(feat.cpu().numpy())

    return np.concatenate(features, axis=0)


def compute_statistics(features: np.ndarray) -> tuple:
    """Calcular media y covarianza de features.

    Returns:
        mu: (2048,), sigma: (2048, 2048)
    """
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2) -> float:
    """Calcular FID entre dos distribuciones.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    # Manejar componentes imaginarias por errores numéricos
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


class FIDCalculator:
    """Calculador de FID con cache de estadísticas reales."""

    def __init__(self, device: torch.device = None, batch_size: int = 64):
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.extractor = InceptionFeatureExtractor(self.device)
        self.real_mu = None
        self.real_sigma = None

    def cache_real_stats(self, real_images: torch.Tensor):
        """Precomputar estadísticas para imágenes reales (una sola vez)."""
        features = compute_features(real_images, self.extractor, self.batch_size)
        self.real_mu, self.real_sigma = compute_statistics(features)
        print(f"Estadísticas reales cacheadas ({features.shape[0]} imágenes)")

    def compute_fid(self, generated_images: torch.Tensor) -> float:
        """Calcular FID entre imágenes generadas y reales cacheadas.

        Args:
            generated_images: tensor (N, 3, H, W) en rango [0, 1]
        """
        if self.real_mu is None:
            raise RuntimeError("Primero cache estadísticas reales con cache_real_stats()")

        features = compute_features(generated_images, self.extractor, self.batch_size)
        gen_mu, gen_sigma = compute_statistics(features)
        return calculate_fid(self.real_mu, self.real_sigma, gen_mu, gen_sigma)
