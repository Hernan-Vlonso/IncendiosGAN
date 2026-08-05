"""Inicialización de pesos para las redes."""

import torch.nn as nn


def init_weights(model: nn.Module):
    """Inicialización de pesos estilo DCGAN (Radford et al. 2016).

    Conv y ConvTranspose: N(0, 0.02)
    BatchNorm: N(1, 0.02), bias=0
    """
    for m in model.modules():
        classname = m.__class__.__name__
        if "Conv" in classname and hasattr(m, "weight"):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif "BatchNorm" in classname and hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif "Linear" in classname and hasattr(m, "weight"):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
