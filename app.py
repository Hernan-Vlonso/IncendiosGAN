#!/usr/bin/env python3
"""Demo Gradio: Generación interactiva de escenarios de incendios forestales.

Interfaz web para la defensa de tesis. Permite ajustar condiciones
meteorológicas y topográficas para generar imágenes de propagación.

Uso: python app.py [--checkpoint PATH] [--share]
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.utils as vutils

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_config
from src.models.generator import Generator


def create_custom_condition(temperature=None, humidity=None,
                            wind_speed=None, wind_dir=None,
                            topography=None):
    """Crear vector de condición personalizado.

    Valores None se tratan como datos faltantes (mask=1).
    Valores continuos en z-score (media 0, std 1).
    """
    condition = np.zeros(32, dtype=np.float32)
    mask = np.zeros(32, dtype=np.float32)

    # Continuos (posiciones 0-3) — valores en z-score
    cont_vals = [temperature, humidity, wind_speed, 0.0]  # sup_total=0
    for i, val in enumerate(cont_vals):
        if val is not None:
            condition[i] = val
        else:
            mask[i] = 1.0
            condition[28 + i] = 1.0  # miss flag

    # Dir_viento (posiciones 14-23)
    wind_cats = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
    if wind_dir is not None and wind_dir != "Desconocido":
        idx = wind_cats.index(wind_dir)
        condition[14 + idx] = 1.0
    else:
        condition[23] = 1.0  # Missing
        mask[14:24] = 1.0

    # Topografía (posiciones 24-27)
    topo_cats = ["Suave", "Irregular", "Abrupta", "Missing"]
    if topography is not None and topography != "Desconocido":
        idx = topo_cats.index(topography)
        condition[24 + idx] = 1.0
    else:
        condition[27] = 1.0  # Missing
        mask[24:28] = 1.0

    return condition, mask


def load_generator(checkpoint_path, cfg, device):
    """Cargar generador desde checkpoint."""
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        img_channels=cfg.model.img_channels,
        features=cfg.model.g_features,
        veg_dim=cfg.model.veg_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema" in ckpt:
        generator.load_state_dict(ckpt["ema"])
    else:
        generator.load_state_dict(ckpt["generator"])
    generator.eval()
    return generator


def build_app(generator, cfg, device):
    """Construir la interfaz Gradio."""

    def generate(temperatura, humedad, vel_viento,
                 temp_missing, hum_missing, viento_missing,
                 dir_viento, topografia, n_samples, seed):
        """Generar imágenes de propagación de incendio."""
        torch.manual_seed(int(seed))

        temp = None if temp_missing else temperatura
        hum = None if hum_missing else humedad
        wind = None if viento_missing else vel_viento
        wind_d = None if dir_viento == "Desconocido" else dir_viento
        topo = None if topografia == "Desconocido" else topografia

        cond, mask = create_custom_condition(
            temperature=temp, humidity=hum, wind_speed=wind,
            wind_dir=wind_d, topography=topo,
        )

        n = int(n_samples)
        z = torch.randn(n, cfg.model.z_dim, device=device)
        cond_t = torch.from_numpy(cond).unsqueeze(0).expand(n, -1).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).expand(n, -1).to(device)
        # veg_profile por defecto: vegetación media uniforme
        veg_t = torch.full((n, cfg.model.veg_dim), 0.1, device=device)

        with torch.no_grad():
            imgs = generator(z, cond_t, mask_t, veg_t)

        # [-1,1] → [0,1] → grid
        imgs = (imgs + 1) / 2
        nrow = min(4, n)
        grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
        grid_np = grid.cpu().permute(1, 2, 0).numpy()
        grid_np = (grid_np * 255).astype(np.uint8)

        # Resumen de condiciones
        info_parts = []
        if temp is not None:
            info_parts.append(f"Temp: {temp:+.1f}σ")
        else:
            info_parts.append("Temp: missing")
        if hum is not None:
            info_parts.append(f"Hum: {hum:+.1f}σ")
        else:
            info_parts.append("Hum: missing")
        if wind is not None:
            info_parts.append(f"Viento: {wind:+.1f}σ")
        else:
            info_parts.append("Viento: missing")
        info_parts.append(f"Dir: {dir_viento}")
        info_parts.append(f"Topo: {topografia}")
        info = " | ".join(info_parts)

        return grid_np, info

    with gr.Blocks(title="IncendiosGANs — Generador de Escenarios") as demo:
        gr.Markdown("# IncendiosGANs — Generador de Escenarios de Incendios Forestales")
        gr.Markdown(
            "Ajusta las condiciones meteorológicas y topográficas para generar "
            "imágenes de propagación de incendios. Los valores continuos están "
            "en z-score (0 = promedio, +2 = muy alto, -2 = muy bajo)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Condiciones Meteorológicas")
                temperatura = gr.Slider(-3, 3, value=0, step=0.1,
                                        label="Temperatura (z-score)")
                temp_missing = gr.Checkbox(label="Temperatura desconocida", value=False)
                humedad = gr.Slider(-3, 3, value=0, step=0.1,
                                    label="Humedad Relativa (z-score)")
                hum_missing = gr.Checkbox(label="Humedad desconocida", value=False)
                vel_viento = gr.Slider(-3, 3, value=0, step=0.1,
                                       label="Velocidad del Viento (z-score)")
                viento_missing = gr.Checkbox(label="Viento desconocido", value=False)

                gr.Markdown("### Condiciones Geográficas")
                dir_viento = gr.Dropdown(
                    choices=["N", "NE", "E", "SE", "S", "SW", "W", "NW",
                             "Calma", "Desconocido"],
                    value="N", label="Dirección del Viento",
                )
                topografia = gr.Dropdown(
                    choices=["Suave", "Irregular", "Abrupta", "Desconocido"],
                    value="Irregular", label="Topografía",
                )

                gr.Markdown("### Parámetros de Generación")
                n_samples = gr.Slider(1, 16, value=4, step=1,
                                      label="Número de muestras")
                seed = gr.Number(value=42, label="Semilla aleatoria", precision=0)

                btn = gr.Button("Generar Escenarios", variant="primary")

            with gr.Column(scale=2):
                output_img = gr.Image(label="Escenarios Generados", type="numpy")
                output_info = gr.Textbox(label="Condiciones utilizadas",
                                         interactive=False)

        btn.click(
            fn=generate,
            inputs=[temperatura, humedad, vel_viento,
                    temp_missing, hum_missing, viento_missing,
                    dir_viento, topografia, n_samples, seed],
            outputs=[output_img, output_info],
        )

        gr.Markdown(
            "---\n"
            "*IncendiosGANs — Conditional DCGAN para simulación de incendios "
            "forestales con datos incompletos. Memoria de Título, UTFSM.*"
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Demo Gradio IncendiosGANs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ruta al checkpoint (default: outputs/checkpoints/checkpoint_final.pt)")
    parser.add_argument("--share", action="store_true",
                        help="Crear link público de Gradio")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or str(cfg.paths.checkpoints / "checkpoint_final.pt")
    print(f"Cargando generador desde: {ckpt_path}")
    print(f"Device: {device}")

    generator = load_generator(ckpt_path, cfg, device)
    demo = build_app(generator, cfg, device)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
