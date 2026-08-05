"""Embedding de condiciones con soporte para datos incompletos (mask)."""

import torch
import torch.nn as nn


class ConditionEmbedding(nn.Module):
    """Red de embedding para vectores de condición con máscara de missingness
    y perfil de vegetación.

    Entrada: [condition(32)*(1-mask) | mask(32) | veg_profile(10)] → embedding(128)

    El perfil de vegetación se añade directamente sin máscara, ya que siempre
    está disponible (se calcula a partir de la superficie quemada por tipo).
    Incluirlo permite al modelo aprender que eucalipto (alta inflamabilidad)
    genera patrones de propagación distintos a vegetación agrícola.
    """

    def __init__(self, condition_dim: int = 32, embed_dim: int = 128,
                 hidden_dim: int = 256, veg_dim: int = 10):
        super().__init__()
        # Input: condition*(1-mask) || mask || veg_profile
        input_dim = condition_dim * 2 + veg_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, condition: torch.Tensor, mask: torch.Tensor,
                veg_profile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            condition: (batch, 32) vector de condición
            mask: (batch, 32) máscara de missingness (1 = dato faltante)
            veg_profile: (batch, 10) perfil de vegetación normalizado

        Returns:
            embedding: (batch, 128)
        """
        masked_condition = condition * (1.0 - mask)
        x = torch.cat([masked_condition, mask, veg_profile], dim=1)
        return self.net(x)
