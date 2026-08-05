#!/usr/bin/env python3
"""Overlay del heatmap generado por Run 48 sobre el área quemada real (dNBR GEE).

Uso:
  python scripts/geovalidation_overlay.py \
    --record-idx 1766 \
    --tif LOS_ROBLES_N.tif \
    --checkpoint outputs/checkpoints/run48/checkpoint_best_final.pt \
    --out outputs/figures/geovalidation/LOS_ROBLES_N_overlay.png \
    --n-samples 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

import rasterio
from rasterio.plot import show as rioshow

from src.config import get_config
from src.models.generator import Generator


def load_generator(checkpoint_path: str, cfg):
    gen = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        img_channels=cfg.model.img_channels,
        features=cfg.model.g_features,
        veg_dim=cfg.model.veg_dim,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("ema", ckpt.get("generator"))
    gen.load_state_dict(state)
    gen.eval()
    return gen


def generate_heatmaps(gen, condition, mask, veg_profile, n_samples: int, device):
    condition = torch.from_numpy(condition).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    veg_profile = torch.from_numpy(veg_profile).unsqueeze(0).to(device)

    condition = condition.expand(n_samples, -1)
    mask = mask.expand(n_samples, -1)
    veg_profile = veg_profile.expand(n_samples, -1)

    z = torch.randn(n_samples, gen.z_dim if hasattr(gen, 'z_dim') else 128, device=device)
    with torch.no_grad():
        imgs = gen(z, condition, mask, veg_profile)

    # [-1,1] → [0,1], promedio de canales → heatmap de probabilidad
    imgs = (imgs.cpu().numpy() + 1.0) / 2.0
    heatmap = imgs.mean(axis=1).mean(axis=0)  # (H, W)
    return heatmap, imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-idx", type=int, required=True)
    parser.add_argument("--tif", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/figures/geovalidation/overlay.png")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--fire-name", type=str, default=None)
    parser.add_argument("--fire-date", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Leer nombre e info del registro
    import pandas as pd
    df = pd.read_csv(cfg.paths.processed_csv)
    rec = df.iloc[args.record_idx]
    fire_name = args.fire_name or rec.get('Nombre incendio', f'Registro {args.record_idx}')
    fire_date = args.fire_date or str(rec.get('Inicio', ''))[:10]
    sup_total = rec.get('Sup_Total', '?')
    dir_viento = rec.get('Dir_viento', '?')
    temperatura = rec.get('Temperatura', '?')
    humedad = rec.get('Humedad_Relativa', '?')

    # Cargar datos procesados
    condition_vectors = np.load(cfg.paths.condition_vectors)
    vegetation_profiles = np.load(cfg.paths.vegetation_profiles)
    masks = np.load(cfg.paths.masks)

    condition = condition_vectors[args.record_idx].astype(np.float32)
    mask = masks[args.record_idx].astype(np.float32)
    veg_profile = vegetation_profiles[args.record_idx].astype(np.float32)

    # Cargar generador
    print(f"Cargando checkpoint: {args.checkpoint}")
    gen = load_generator(args.checkpoint, cfg).to(device)

    # Generar heatmaps
    print(f"Generando {args.n_samples} realizaciones...")
    heatmap, all_imgs = generate_heatmaps(gen, condition, mask, veg_profile,
                                           args.n_samples, device)

    # Leer TIF con área quemada real
    with rasterio.open(args.tif) as src:
        burned_data = src.read()   # (bands, H, W)
        bounds = src.bounds
        tif_shape = (src.height, src.width)

    # Máscara binaria: cualquier canal > 0 = quemado
    burned_mask = (burned_data.max(axis=0) > 50).astype(np.float32)

    # Redimensionar heatmap al tamaño del TIF para overlay
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = np.array(
        heatmap_img.resize((tif_shape[1], tif_shape[0]), Image.BILINEAR)
    ) / 255.0

    # Imagen satelital de fondo (RGB del TIF si tiene 3 bandas, sino gris)
    if burned_data.shape[0] >= 3:
        bg = np.stack([burned_data[0], burned_data[1], burned_data[2]], axis=-1)
    else:
        bg = np.stack([burned_data[0]] * 3, axis=-1)
    bg = (bg / bg.max() * 255).astype(np.uint8) if bg.max() > 0 else bg

    # Calcular centroide del heatmap para flecha direccional
    H, W = heatmap.shape
    yy = np.arange(H) - (H - 1) / 2.0
    xx = np.arange(W) - (W - 1) / 2.0
    mass = heatmap.sum().clip(min=1e-6)
    cy = (heatmap * yy[:, None]).sum() / mass
    cx = (heatmap * xx[None, :]).sum() / mass
    # Normalizar para mostrar como flecha desde el centro
    arrow_scale = (H / 2) * 0.6
    norm = max(np.sqrt(cx**2 + cy**2), 1e-6)
    ax_cy = cy / norm * arrow_scale
    ax_cx = cx / norm * arrow_scale

    # Normalizar heatmap al rango real para mostrar gradiente
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_norm = (heatmap - h_min) / max(h_max - h_min, 1e-6)
    heatmap_resized_norm = np.array(
        Image.fromarray((heatmap_norm * 255).astype(np.uint8))
        .resize((tif_shape[1], tif_shape[0]), Image.BILINEAR)
    ) / 255.0

    # Plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Área quemada real (dNBR)
    axes[0].imshow(burned_mask, cmap='Reds', interpolation='nearest')
    axes[0].set_title(f'Área quemada real (dNBR > 0.35)\n{fire_name} — {sup_total} ha', fontsize=11)
    axes[0].axis('off')

    # Mancha de fuego: solo píxeles sobre el percentil 60 (zona de mayor probabilidad)
    threshold = np.percentile(heatmap_norm, 60)
    fire_blob = np.ma.masked_where(heatmap_norm < threshold, heatmap_norm)

    fire_blob_resized = np.ma.masked_where(
        np.array(Image.fromarray((heatmap_norm * 255).astype(np.uint8))
                 .resize((tif_shape[1], tif_shape[0]), Image.BILINEAR)) / 255.0 < threshold,
        np.array(Image.fromarray((heatmap_norm * 255).astype(np.uint8))
                 .resize((tif_shape[1], tif_shape[0]), Image.BILINEAR)) / 255.0
    )

    cmap_fire = plt.cm.Oranges
    cmap_fire.set_bad(color='white')  # píxeles enmascarados = blanco

    # Panel 2: fondo blanco, solo mancha naranja + flecha
    axes[1].set_facecolor('white')
    axes[1].imshow(fire_blob, cmap=cmap_fire, vmin=threshold, vmax=1,
                   interpolation='bilinear')
    cx_img, cy_img = W / 2, H / 2
    axes[1].annotate('', xy=(cx_img + ax_cx, cy_img + ax_cy),
                     xytext=(cx_img, cy_img),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    axes[1].plot(cx_img, cy_img, 'b+', markersize=8)
    axes[1].set_title(f'Probabilidad de quema Run 48 (top 40%)\nDir_viento={dir_viento}, T={temperatura}°C, HR={humedad}%\n(flecha azul = dirección centroide)', fontsize=10)
    axes[1].axis('off')

    # Panel 3: overlay — modelo naranja sobre real rojo, fondo blanco
    axes[2].set_facecolor('white')
    axes[2].imshow(burned_mask, cmap='Reds', vmin=0, vmax=1,
                   interpolation='nearest')
    axes[2].imshow(fire_blob_resized, cmap=cmap_fire, vmin=threshold, vmax=1,
                   alpha=0.6, interpolation='bilinear')
    axes[2].set_title('Overlay: real (rojo) + modelo (naranja)\n← W   E →   ↑ N   S ↓', fontsize=11)
    axes[2].axis('off')

    plt.suptitle(f'Validación geoespacial — {fire_name} ({fire_date})\nRun 48 · checkpoint_best_final.pt',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Guardado en: {out_path}")

    # También guardar el heatmap solo (64x64 original)
    heatmap_out = out_path.parent / (out_path.stem + "_heatmap64.png")
    heatmap_img_color = plt.cm.hot(heatmap)
    plt.imsave(str(heatmap_out), heatmap_img_color)
    print(f"Heatmap 64×64: {heatmap_out}")


if __name__ == "__main__":
    main()
