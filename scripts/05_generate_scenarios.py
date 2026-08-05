#!/usr/bin/env python3
"""Script 05: Generar escenarios de incendios con el modelo entrenado.

Permite generar imágenes de propagación condicionadas a parámetros específicos,
incluyendo escenarios con datos incompletos.

Uso: python scripts/05_generate_scenarios.py --checkpoint PATH [--n-samples N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import get_config
from src.models.generator import Generator
from src.visualization.condition_explorer import sweep_masking

EXPOSICION_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Plano", "Missing"]
DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
TOPOGRAFIA_CATS = ["Suave", "Irregular", "Abrupta", "Missing"]


def create_custom_condition(temperature=None, humidity=None,
                            wind_speed=None, sup_total=None,
                            exposicion=None, wind_dir=None,
                            topography=None):
    """Crear vector de condición personalizado (32 dims).

    Valores None se tratan como datos faltantes (mask=1).

    Dims 0-3:   continuos (Temperatura, Humedad, Vel_Viento, Sup_Total)
    Dims 4-13:  Exposicion (one-hot 10 cats)
    Dims 14-23: Dir_viento (one-hot 10 cats)
    Dims 24-27: Topografia (one-hot 4 cats)
    Dims 28-31: miss_flags para los 4 continuos
    """
    condition = np.zeros(32, dtype=np.float32)
    mask = np.zeros(32, dtype=np.float32)

    # Continuos (0-3)
    for i, val in enumerate([temperature, humidity, wind_speed, sup_total]):
        if val is not None:
            condition[i] = val
        else:
            mask[i] = 1.0
            condition[28 + i] = 1.0

    # Exposicion (dims 4-13)
    if exposicion is not None and exposicion in EXPOSICION_CATS:
        condition[4 + EXPOSICION_CATS.index(exposicion)] = 1.0
    else:
        condition[4 + EXPOSICION_CATS.index("Missing")] = 1.0
        mask[4:14] = 1.0

    # Dir_viento (dims 14-23)
    if wind_dir is not None and wind_dir in DIR_VIENTO_CATS:
        condition[14 + DIR_VIENTO_CATS.index(wind_dir)] = 1.0
    else:
        condition[14 + DIR_VIENTO_CATS.index("Missing")] = 1.0
        mask[14:24] = 1.0

    # Topografia (dims 24-27)
    if topography is not None and topography in TOPOGRAFIA_CATS:
        condition[24 + TOPOGRAFIA_CATS.index(topography)] = 1.0
    else:
        condition[24 + TOPOGRAFIA_CATS.index("Missing")] = 1.0
        mask[24:28] = 1.0

    return condition, mask


@torch.no_grad()
def sweep_wind_direction(generator, base_condition, base_mask, z_fixed,
                         device, veg, output_path):
    """Generar una imagen por cada dirección de viento con z y condiciones base fijos."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma"]
    n = len(directions)

    conditions = np.tile(base_condition, (n, 1))
    masks = np.tile(base_mask, (n, 1))

    # Sobreescribir bloque Dir_viento con el one-hot correcto
    for i, d in enumerate(directions):
        conditions[i, 14:24] = 0.0
        masks[i, 14:24] = 0.0
        conditions[i, 14 + DIR_VIENTO_CATS.index(d)] = 1.0

    z = z_fixed.expand(n, -1)
    cond_t = torch.from_numpy(conditions.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(masks.astype(np.float32)).to(device)
    veg_t = veg.expand(n, -1)

    imgs = generator(z, cond_t, mask_t, veg_t)
    imgs = (imgs.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    for ax, img, d in zip(axes, imgs, directions):
        ax.imshow(img.cpu().permute(1, 2, 0).squeeze().numpy(),
                  cmap="hot" if img.shape[0] == 1 else None)
        ax.set_title(d, fontsize=11)
        ax.axis("off")
    plt.suptitle("Sweep Dir_viento — condiciones base fijas, mismo z\n"
                 "(viento DESDE la dirección indicada; fuego se propaga en dirección opuesta)", fontsize=11)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardado: {output_path}")
    return imgs.cpu()


@torch.no_grad()
def compare_opposite_winds(generator, base_condition, base_mask, z_fixed,
                            device, veg, output_path, n_samples=8):
    """Comparar N vs S (y E vs W) con múltiples z para verificar que el efecto es consistente."""
    pairs = [("N", "S"), ("E", "W"), ("NE", "SW")]
    n = n_samples

    z = torch.randn(n, generator.z_dim if hasattr(generator, "z_dim") else 128, device=device)

    fig, axes = plt.subplots(len(pairs) * 2, n, figsize=(2 * n, 3 * len(pairs) * 2))

    for row_pair, (dir_a, dir_b) in enumerate(pairs):
        for col, (direction, row_offset) in enumerate([(dir_a, 0), (dir_b, 1)]):
            cond = base_condition.copy()
            msk = base_mask.copy()
            cond[14:24] = 0.0
            msk[14:24] = 0.0
            cond[14 + DIR_VIENTO_CATS.index(direction)] = 1.0

            cond_t = torch.from_numpy(np.tile(cond, (n, 1)).astype(np.float32)).to(device)
            mask_t = torch.from_numpy(np.tile(msk, (n, 1)).astype(np.float32)).to(device)
            veg_t = veg.expand(n, -1)

            imgs = generator(z, cond_t, mask_t, veg_t)
            imgs = (imgs.clamp(-1, 1) + 1) / 2

            row_idx = row_pair * 2 + row_offset
            for col_idx in range(n):
                ax = axes[row_idx, col_idx]
                img = imgs[col_idx].cpu().permute(1, 2, 0).squeeze().numpy()
                ax.imshow(img, cmap="hot" if imgs.shape[1] == 1 else None)
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(direction, fontsize=12, rotation=0, labelpad=30)

    plt.suptitle("Comparación direcciones opuestas (mismo z, n=8)", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generar escenarios de incendios")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Cargar generador
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        img_channels=cfg.model.img_channels,
        features=cfg.model.g_features,
        veg_dim=cfg.model.veg_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        generator.load_state_dict(ckpt["ema"])
        print("Usando pesos EMA")
    else:
        generator.load_state_dict(ckpt["generator"])
    generator.eval()

    output_dir = cfg.paths.figures / "scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    default_veg = torch.full((1, cfg.model.veg_dim), 0.1, device=device)
    z_fixed = torch.randn(1, cfg.model.z_dim, device=device)

    print("=" * 60)
    print("GENERACIÓN DE ESCENARIOS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Escenario 1: Condiciones completas — caluroso, viento N, abrupto
    # ------------------------------------------------------------------
    print("\n1. Escenario: Día caluroso + viento N + terreno abrupto")
    cond, mask = create_custom_condition(
        temperature=2.0, humidity=-1.5, wind_speed=1.5,
        exposicion="S", wind_dir="N", topography="Abrupta",
    )
    z = torch.randn(args.n_samples, cfg.model.z_dim, device=device)
    cond_t = torch.from_numpy(cond).unsqueeze(0).expand(args.n_samples, -1).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).expand(args.n_samples, -1).to(device)
    veg_t = default_veg.expand(args.n_samples, -1)

    with torch.no_grad():
        imgs = generator(z, cond_t, mask_t, veg_t)
        imgs = (imgs.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        if i < args.n_samples:
            img = imgs[i].cpu().permute(1, 2, 0).squeeze().numpy()
            ax.imshow(img, cmap="hot" if imgs.shape[1] == 1 else None)
        ax.axis("off")
    plt.suptitle("Caluroso + Viento N + Terreno Abrupto (n=16 z distintos)", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(output_dir / "scenario_hot_windy_N.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardado: {output_dir / 'scenario_hot_windy_N.png'}")

    # ------------------------------------------------------------------
    # Escenario 2: Solo temperatura conocida (resto missing)
    # ------------------------------------------------------------------
    print("\n2. Escenario: Solo temperatura conocida (resto missing)")
    cond, mask = create_custom_condition(temperature=1.5)
    cond_t = torch.from_numpy(cond).unsqueeze(0).expand(args.n_samples, -1).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).expand(args.n_samples, -1).to(device)

    with torch.no_grad():
        imgs = generator(z, cond_t, mask_t, veg_t)
        imgs = (imgs.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        if i < args.n_samples:
            img = imgs[i].cpu().permute(1, 2, 0).squeeze().numpy()
            ax.imshow(img, cmap="hot" if imgs.shape[1] == 1 else None)
        ax.axis("off")
    plt.suptitle("Solo Temperatura Conocida — Alta Incertidumbre", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(output_dir / "scenario_incomplete.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardado: {output_dir / 'scenario_incomplete.png'}")

    # ------------------------------------------------------------------
    # Escenario 3: Sweep de temperatura (-2 a 2 z-score) — z fijo
    # ------------------------------------------------------------------
    print("\n3. Sweep de temperatura (-2 a 2 z-score) con z fijo")
    cond_base, mask_base = create_custom_condition(
        temperature=0, humidity=0, wind_speed=0,
        exposicion="S", wind_dir="N", topography="Irregular",
    )
    temps = np.linspace(-2, 2, 8)
    conditions = np.tile(cond_base, (len(temps), 1))
    conditions[:, 0] = temps
    masks = np.tile(mask_base, (len(temps), 1))
    z_exp = z_fixed.expand(len(temps), -1)
    cond_t = torch.from_numpy(conditions.astype(np.float32)).to(device)
    mask_t = torch.from_numpy(masks.astype(np.float32)).to(device)
    veg_t2 = default_veg.expand(len(temps), -1)

    with torch.no_grad():
        imgs = generator(z_exp, cond_t, mask_t, veg_t2)
        imgs = (imgs.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(1, len(temps), figsize=(2.5 * len(temps), 3))
    for ax, img, t in zip(axes, imgs, temps):
        ax.imshow(img.cpu().permute(1, 2, 0).squeeze().numpy(),
                  cmap="hot" if img.shape[0] == 1 else None)
        ax.set_title(f"T={t:.1f}σ", fontsize=10)
        ax.axis("off")
    plt.suptitle("Sweep Temperatura (z fijo, viento N, terreno irregular)", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(output_dir / "sweep_temperature.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardado: {output_dir / 'sweep_temperature.png'}")

    # ------------------------------------------------------------------
    # Escenario 4: Sweep de Dir_viento — VALIDACIÓN FÍSICA PRINCIPAL
    # ------------------------------------------------------------------
    print("\n4. Sweep Dir_viento — validación física principal")
    cond_base, mask_base = create_custom_condition(
        temperature=1.5, humidity=-1.0, wind_speed=1.5,
        exposicion="S", topography="Irregular",
        wind_dir="N",  # se sobreescribirá en el sweep
    )
    sweep_wind_direction(
        generator, cond_base, mask_base, z_fixed, device, default_veg,
        output_dir / "sweep_wind_direction.png",
    )

    # ------------------------------------------------------------------
    # Escenario 5: Comparación direcciones opuestas (N vs S, E vs W)
    # ------------------------------------------------------------------
    print("\n5. Comparación N vs S, E vs W, NE vs SW")
    compare_opposite_winds(
        generator, cond_base, mask_base, z_fixed, device, default_veg,
        output_dir / "comparison_opposite_winds.png",
        n_samples=8,
    )

    # ------------------------------------------------------------------
    # Escenario 6: Masking progresivo
    # ------------------------------------------------------------------
    print("\n6. Masking progresivo")
    cond_full, mask_full = create_custom_condition(
        temperature=1.0, humidity=-0.5, wind_speed=1.0,
        exposicion="SW", wind_dir="SW", topography="Abrupta",
    )
    _, fig = sweep_masking(
        generator, cond_full, mask_full,
        dims_to_mask=[0, 1, 2, list(range(14, 24))],
        z_dim=cfg.model.z_dim, device=device,
    )
    fig.savefig(str(output_dir / "sweep_masking.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardado: {output_dir / 'sweep_masking.png'}")

    print(f"\nEscenarios guardados en: {output_dir}")


if __name__ == "__main__":
    main()
