#!/usr/bin/env python3
"""Diagnóstico 6: Test unitario del gradiente del centroid loss.

Para cada dirección, crea una imagen sintética con fuego centrado
y verifica que el gradiente del centroid loss empuja el fuego
en la dirección correcta (downwind).
"""

import sys
import math
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config

_SQRT2 = math.sqrt(2) / 2.0
_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_DIR_TARGET = [
    ( 0.0,   +1.0  ),
    (-_SQRT2,+_SQRT2),
    (-1.0,    0.0  ),
    (-_SQRT2,-_SQRT2),
    ( 0.0,   -1.0  ),
    (+_SQRT2,-_SQRT2),
    (+1.0,    0.0  ),
    (+_SQRT2,+_SQRT2),
]
_DIR_VIENTO_START = 14


# Centroid loss actual del trainer (run34 formulation)
def centroid_direction_loss(fake_imgs, condition):
    B, C, H, W = fake_imgs.shape
    targets = torch.tensor(_DIR_TARGET, dtype=torch.float32, device=fake_imgs.device)
    wind_onehot = condition[:, 14:22]
    valid = wind_onehot.sum(dim=1) > 0.5
    if not valid.any():
        return fake_imgs.new_zeros(()).requires_grad_(False)
    fire = (fake_imgs.mean(dim=1) + 1.0) / 2.0
    yy = torch.arange(H, dtype=fake_imgs.dtype, device=fake_imgs.device) - (H-1)/2.0
    xx = torch.arange(W, dtype=fake_imgs.dtype, device=fake_imgs.device) - (W-1)/2.0
    mass = fire.sum(dim=(1,2)).clamp(min=1e-6)
    cy = (fire * yy.view(1,H,1)).sum(dim=(1,2)) / mass
    cx = (fire * xx.view(1,1,W)).sum(dim=(1,2)) / mass
    v = torch.stack([cx, cy], dim=1)
    disp_weight = v.norm(dim=1).clamp(max=H/2.0) / (H/2.0)
    dir_idx = wind_onehot.argmax(dim=1)
    t = targets[dir_idx]
    v_unit = v / (v.norm(dim=1, keepdim=True) + 1e-6)
    cos_sim = (v_unit * t).sum(dim=1)
    loss_per_sample = (1.0 - cos_sim) * disp_weight
    return loss_per_sample[valid].mean()


def make_condition(d, H, device):
    """Condition vector sintético con solo Dir_viento[d]=1."""
    cond = torch.zeros(1, 32, device=device)
    cond[0, 14 + d] = 1.0
    return cond


def test_gradient_direction():
    """
    Para una imagen con fuego desplazado al NORTE (cy < 0),
    el gradiente debería empujar hacia el downwind de cada dirección.
    Mide: grad · t (positivo = gradiente apunta en dirección correcta).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W, C = 64, 64, 3

    print("Test 1: Imagen con fuego centrado (cx=0, cy=0)")
    print("  Esperado: loss ~0 (no hay displacement = disp_weight=0)\n")

    # Imagen con fuego exactamente centrado
    img_center = torch.zeros(1, C, H, W, device=device)
    img_center[0, :, H//2, W//2] = 1.0  # pixel central

    for d, name in enumerate(_DIRS):
        cond = make_condition(d, H, device)
        loss = centroid_direction_loss(img_center, cond)
        print(f"  {name:>4}: loss={loss.item():.4f}")

    print("\nTest 2: Imagen con fuego desplazado al OESTE (cx<0, cy~0)")
    print("  Esperado: loss pequeño para E (fuego va al oeste = correcto)")
    print("            loss grande para W (fuego va al oeste = incorrecto)\n")

    # Imagen con fuego desplazado al OESTE
    img_west = torch.full((1, C, H, W), -1.0, device=device)
    cx_target = -H // 4  # desplazamiento de H/4 hacia la izquierda
    x_center = W // 2 + cx_target
    r = 5
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy**2 + dx**2 <= r**2:
                y = H//2 + dy
                x = x_center + dx
                if 0 <= y < H and 0 <= x < W:
                    img_west[0, :, y, x] = 1.0

    for d, name in enumerate(_DIRS):
        cond = make_condition(d, H, device)
        loss = centroid_direction_loss(img_west, cond)
        print(f"  {name:>4}: loss={loss.item():.4f}")

    print("\nTest 3: Gradiente desde imagen centrada — ¿empuja en dirección correcta?")
    print("  Medida: grad·t (positivo = OK, negativo = bug)\n")
    print(f"  {'Dir':>4} {'grad·t':>10} {'OK?':>6}")
    print("  " + "-" * 25)

    # Para gradiente con fuego centrado: loss = 0 si disp_weight=0
    # Necesitamos pequeño desplazamiento inicial para que haya gradiente
    # Usar imagen con ruido gaussiano pequeño
    torch.manual_seed(42)
    img_noisy = torch.randn(1, C, H, W, device=device) * 0.1  # fuego casi uniforme/centrado

    for d, name in enumerate(_DIRS):
        img = img_noisy.clone().requires_grad_(True)
        cond = make_condition(d, H, device)
        loss = centroid_direction_loss(img, cond)
        loss.backward()

        # Calcular cx, cy del gradiente (suma ponderada por posición)
        # El gradiente en cada pixel indica hacia dónde moverlo
        grad = img.grad[0].mean(dim=0)  # [H, W]
        yy = torch.arange(H, dtype=torch.float32, device=device) - (H-1)/2.0
        xx = torch.arange(W, dtype=torch.float32, device=device) - (W-1)/2.0
        # Centroide del gradiente (negativo = dónde queremos llevar el fuego)
        # El loss se minimiza moviendo el centroide hacia t
        # grad < 0 en posiciones target = queremos poner fuego ahí
        neg_grad = -grad
        mass_g = neg_grad.clamp(min=0).sum().clamp(min=1e-6)
        gcy = (neg_grad.clamp(min=0) * yy.view(H,1)).sum() / mass_g
        gcx = (neg_grad.clamp(min=0) * xx.view(1,W)).sum() / mass_g

        tx, ty = _DIR_TARGET[d]
        dot = gcx.item() * tx + gcy.item() * ty
        ok = "OK" if dot > 0 else "CHECK"
        print(f"  {name:>4} {dot:>+10.4f}  {ok}")


if __name__ == "__main__":
    test_gradient_direction()
