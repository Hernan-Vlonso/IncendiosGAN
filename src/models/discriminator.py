"""Discriminator con projection conditioning y spectral normalization."""

import math
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from src.models.conditioning import ConditionEmbedding


class Discriminator(nn.Module):
    """Discriminator condicional con projection (Miyato et al. 2018).

    Arquitectura:
    img(3, 64, 64) → SN-Conv(3→64) → SN-Conv(64→128) → SN-Conv(128→256) → SN-Conv(256→512)
    → GlobalAvgPool → Linear(512→1) + scaled projection(cond_embed · h)

    Spectral normalization en todas las capas convolucionales.
    LeakyReLU(0.2).
    """

    def __init__(self, img_channels: int = 3, condition_dim: int = 32,
                 cond_embed_dim: int = 128, cond_hidden_dim: int = 256,
                 features: list = None, veg_dim: int = 10):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        # Embedding de condiciones + perfil de vegetación
        self.cond_embedding = ConditionEmbedding(
            condition_dim=condition_dim,
            embed_dim=cond_embed_dim,
            hidden_dim=cond_hidden_dim,
            veg_dim=veg_dim,
        )

        # Bloques convolucionales con spectral norm
        layers = []

        # Primera capa: sin normalización
        layers.append(spectral_norm(
            nn.Conv2d(img_channels, features[0], 4, 2, 1, bias=False)
        ))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Capas intermedias: SN ya controla la norma espectral, sin normalización adicional
        # (InstanceNorm causaba estadísticas ruidosas que desestabilizaban el entrenamiento)
        for i in range(len(features) - 1):
            layers.append(spectral_norm(
                nn.Conv2d(features[i], features[i + 1], 4, 2, 1, bias=False)
            ))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.conv = nn.Sequential(*layers)

        # Global average pooling → vector de features[−1] dims (no 8192)
        self.feat_dim = features[-1]

        # Cabeza lineal (output escalar) — +1 por minibatch stddev
        self.linear = spectral_norm(nn.Linear(self.feat_dim + 1, 1))

        # Projection: cond_embed → features[-1] para dot product escalado
        self.projection = spectral_norm(
            nn.Linear(cond_embed_dim, self.feat_dim, bias=False)
        )

        # Factor de escala para el dot product (1/sqrt(dim))
        self.proj_scale = 1.0 / math.sqrt(self.feat_dim)

        # Cabeza auxiliar de clasificación de dirección de viento (AC-GAN, Run 44+)
        # 8 clases = N, NE, E, SE, S, SW, W, NW (Calma y Missing se filtran en trainer
        # con `valid = (dir_labels < 8) & has_dir` antes de calcular el CE loss).
        # Sin spectral norm: es un clasificador supervisado, no parte del juego adversarial.
        self.dir_head = nn.Linear(self.feat_dim, 8)

    def extract_features(self, img: torch.Tensor) -> list:
        """Devuelve feature maps intermedios tras cada bloque conv+LeakyReLU.

        Usado para feature matching loss: fuerza al generador a producir
        activaciones con estadísticas similares a las imágenes reales.

        Returns:
            Lista de 4 tensores (B, C, H, W) a escalas 32, 16, 8, 4.
        """
        features = []
        x = img
        for module in self.conv:
            x = module(x)
            if isinstance(module, nn.LeakyReLU):
                features.append(x)
        return features

    def forward(self, img: torch.Tensor, condition: torch.Tensor,
                mask: torch.Tensor, veg_profile: torch.Tensor,
                return_dir: bool = False):
        """
        Args:
            img: (batch, 3, 64, 64) imagen en rango [-1, 1]
            condition: (batch, 32) vector de condición
            mask: (batch, 32) máscara
            veg_profile: (batch, 10) perfil de vegetación
            return_dir: si True, retorna también logits de dirección de viento

        Returns:
            score: (batch, 1) puntaje de realismo
            dir_logits: (batch, 8) logits de dirección [solo si return_dir=True]
        """
        h = self.conv(img)
        h = h.mean(dim=[2, 3])

        std = h.std(dim=0, unbiased=False).mean().expand(h.size(0), 1)
        h_aug = torch.cat([h, std], dim=1)
        output = self.linear(h_aug)

        cond_embed = self.cond_embedding(condition, mask, veg_profile)
        proj = self.projection(cond_embed)
        output = output + self.proj_scale * (proj * h).sum(dim=1, keepdim=True)

        if return_dir:
            dir_logits = self.dir_head(h)
            return output, dir_logits
        return output
