"""Conditional DCGAN Generator con Conditional Batch Normalization (CBN)."""

import torch
import torch.nn as nn

from src.models.conditioning import ConditionEmbedding


class CBNBlock(nn.Module):
    """Bloque de upsampling con Conditional Batch Normalization.

    Reemplaza BatchNorm2d(affine=True) fijo por modulación gamma/beta
    predicha desde cond_embed en cada escala espacial.

    Formulación: y = (1 + gamma) * BN(x) + beta
    - BN(affine=False): normaliza sin parámetros aprendibles propios
    - gamma, beta: funciones lineales de cond_embed por bloque
    - Inicialización cerca de 0 → parte cerca de identidad, sin desestabilizar

    Cada bloque tiene sus propios parámetros CBN (no compartidos).
    """

    def __init__(self, in_ch: int, out_ch: int, cond_embed_dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, affine=False)
        self.act  = nn.ReLU(True)

        # Predecir gamma y beta desde cond_embed → 2 * out_ch valores
        self.cbn_linear = nn.Linear(cond_embed_dim, 2 * out_ch)
        # Inicializar cerca de identidad: gamma≈0, beta≈0
        nn.init.normal_(self.cbn_linear.weight, std=0.02)
        nn.init.zeros_(self.cbn_linear.bias)

    def forward(self, x: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        # gamma, beta: (B, out_ch) → (B, out_ch, 1, 1) para broadcast espacial
        gamma, beta = self.cbn_linear(cond_embed).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        x = (1 + gamma) * x + beta
        return self.act(x)


class Generator(nn.Module):
    """Generator condicional tipo DCGAN con CBN en bloques de upsampling.

    Arquitectura:
    z(128) + cond_embed(128) → Linear(256, 512*4*4) → Reshape(512, 4, 4)
    → CBNBlock(512→256, 4→8) → CBNBlock(256→128, 8→16)
    → CBNBlock(128→64, 16→32) → ConvT(64→3, 32→64) → Tanh

    Cada CBNBlock predice gamma_i y beta_i desde cond_embed, modulando
    la normalización en cada escala espacial (no solo en la entrada).
    """

    def __init__(self, z_dim: int = 128, condition_dim: int = 32,
                 cond_embed_dim: int = 128, cond_hidden_dim: int = 256,
                 img_channels: int = 3, features: list = None,
                 veg_dim: int = 10):
        super().__init__()

        if features is None:
            features = [512, 256, 128, 64]

        self.z_dim = z_dim
        self.cond_embed_dim = cond_embed_dim

        # Embedding de condiciones + perfil de vegetación (compartido)
        self.cond_embedding = ConditionEmbedding(
            condition_dim=condition_dim,
            embed_dim=cond_embed_dim,
            hidden_dim=cond_hidden_dim,
            veg_dim=veg_dim,
        )

        # Proyección z + cond_embed → feature map inicial (BN1d está bien aquí:
        # no es espacial y cond_embed ya está concatenado con z)
        self.project = nn.Sequential(
            nn.Linear(z_dim + cond_embed_dim, features[0] * 4 * 4),
            nn.BatchNorm1d(features[0] * 4 * 4),
            nn.ReLU(True),
        )

        self.init_channels = features[0]

        # Bloques CBN de upsampling — ModuleList porque forward necesita pasar
        # cond_embed a cada bloque (no compatible con nn.Sequential)
        self.blocks = nn.ModuleList([
            CBNBlock(features[i], features[i + 1], cond_embed_dim)
            for i in range(len(features) - 1)
        ])

        # Capa final: features[-1] → img_channels (sin BN, solo Tanh)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features[-1], img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor,
                mask: torch.Tensor, veg_profile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, 128) ruido latente
            condition: (batch, 32) vector de condición
            mask: (batch, 32) máscara de missingness
            veg_profile: (batch, 10) perfil de vegetación

        Returns:
            imagen: (batch, 3, 64, 64) en rango [-1, 1]
        """
        cond_embed = self.cond_embedding(condition, mask, veg_profile)

        # Proyectar z + cond_embed a feature map inicial
        x = self.project(torch.cat([z, cond_embed], dim=1))
        x = x.view(-1, self.init_channels, 4, 4)

        # Upsampling con CBN: cond_embed modula cada escala
        for block in self.blocks:
            x = block(x, cond_embed)

        return self.final(x)
