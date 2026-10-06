#!/usr/bin/env python3
"""Demo Gradio: escenarios de propagación de incendios sobre casos históricos reales.

Parte de un incendio registrado por CONAF (con su vegetación y condiciones reales) y
permite modificar viento, clima y superficie para generar mapas de calor de propagación.

Uso: python app.py [--checkpoint PATH] [--share] [--port N]
Requiere demo/ (ver scripts/11_export_demo_assets.py).
"""

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.veg_utils import composite_fire_on_veg, veg_legend_patches
from src.config import get_config
from src.models.generator import Generator
from src.visualization.fire_render import DOWNWIND, direction_error_deg, fire_intensity

DEMO_DIR = ROOT / "demo"
WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma"]
TOPOGRAPHIES = ["Suave", "Irregular", "Abrupta"]
UNKNOWN = "Desconocida"
LOG_HA_RANGE = (-1.0, 5.3)
# Error angular (media, desv. est.) en grados por dirección: validación CONAF run48,
# 10 incendios reales por dirección x 30 realizaciones (docs/defensa_presentacion.tex)
VALIDATION_ERR_DEG = {"N": (15.1, 19.8), "NE": (17.6, 21.0), "E": (7.5, 8.8), "SE": (34.2, 18.7),
                      "S": (2.1, 33.2), "SW": (1.3, 15.6), "W": (6.7, 15.8), "NW": (11.3, 24.4)}
COMPASS = {"N": "S", "NE": "SW", "E": "W", "SE": "NW", "S": "N", "SW": "NE", "W": "E", "NW": "SE"}
BG, TXT = "#111122", "#FFFFFF"


def load_generator(checkpoint_path, cfg, device):
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
    state = ckpt["ema"] if "ema" in ckpt else ckpt.get("generator", ckpt)
    generator.load_state_dict(state)
    generator.eval()
    return generator


def set_onehot(cond, mask, start, size, index, missing):
    cond[start:start + size] = 0.0
    cond[start + index] = 1.0
    mask[start:start + size] = float(missing)


def build_condition(meta, case, values, missing, ha, wind_dir, topography):
    """Vector de condición y máscara con la misma codificación que el entrenamiento."""
    cond = np.array(case["condition"], dtype=np.float32)
    mask = np.array(case["mask"], dtype=np.float32)
    mean, scale = meta["scaler_mean"], meta["scaler_scale"]

    for i, (value, is_missing) in enumerate(zip(values, missing)):
        cond[i] = meta["imputed_z"][i] if is_missing else (value - mean[i]) / scale[i]
        cond[28 + i] = mask[i] = mask[28 + i] = float(is_missing)
    cond[3] = (np.log1p(ha) - mean[3]) / scale[3]

    wind_known = wind_dir != UNKNOWN
    set_onehot(cond, mask, 14, 10, WIND_DIRS.index(wind_dir) if wind_known else 9, not wind_known)
    topo_known = topography != UNKNOWN
    set_onehot(cond, mask, 24, 4, TOPOGRAPHIES.index(topography) if topo_known else 3, not topo_known)
    return cond, mask


def render_samples(fires, veg, wind_dir, errors):
    n = len(fires)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig = Figure(figsize=(3.2 * ncols, 3.4 * nrows), facecolor=BG)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= n:
            continue
        fire = fires[k]
        fire_norm = np.clip(fire / max(fire.max() * 0.95, 0.05), 0, 1)
        ax.imshow(composite_fire_on_veg(fire_norm, veg), origin="upper", interpolation="bilinear")
        title = f"Muestra {k + 1}"
        if errors[k] is not None:
            title += f"  ·  desv. {errors[k]:.0f}°"
        ax.set_title(title, fontsize=10, fontweight="bold", color=TXT, pad=5)
        if wind_dir in DOWNWIND:
            dx, dy = DOWNWIND[wind_dir]
            h, w = fire.shape
            scale = h * 0.28
            ax.annotate("", xy=(w / 2 + dx * scale, h / 2 + dy * scale), xytext=(w / 2, h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="cyan", lw=3.0, mutation_scale=20), zorder=5)
        if k == 0:
            patches = veg_legend_patches(veg, top_n=2)
            if patches:
                ax.legend(handles=patches, loc="lower right", fontsize=6, framealpha=0.55,
                          facecolor="#111", edgecolor="none", labelcolor="white", handlelength=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.05, hspace=0.12)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def confidence_text(wind_dir, errors):
    if wind_dir not in DOWNWIND:
        return ("Sin dirección de viento definida: no hay una dirección esperada contra la cual "
                "verificar la propagación.")
    valid = [e for e in errors if e is not None]
    batch = f"{np.median(valid):.0f}°" if valid else "n/d"
    mean_err, std_err = VALIDATION_ERR_DEG[wind_dir]
    level = "alta" if mean_err < 10 else "media" if mean_err < 20 else "baja"
    return (f"**Dirección downwind esperada:** hacia el {COMPASS[wind_dir]} (flecha cian)  \n"
            f"**Desviación angular mediana de estas muestras:** {batch}  \n"
            f"**Confianza del modelo para viento {wind_dir}:** {level} "
            f"(error angular en validación CONAF: {mean_err:.1f}° ± {std_err:.1f}°)")


def case_defaults(meta, case):
    ranges = meta["ranges"]
    mid = {k: round((lo + hi) / 2) for k, (lo, hi) in ranges.items()}
    return {
        "temp": case["temp"] if case["temp"] is not None else mid["temp"],
        "hum": case["hum"] if case["hum"] is not None else mid["hum"],
        "wind_speed": case["wind_speed"] if case["wind_speed"] is not None else mid["wind_speed"],
        "miss_t": case["temp"] is None,
        "miss_h": case["hum"] is None,
        "miss_w": case["wind_speed"] is None,
        "log_ha": float(np.clip(np.log10(case["ha"]), *LOG_HA_RANGE)),
        "wind_dir": case["wind_dir"] if case["wind_dir"] in WIND_DIRS else UNKNOWN,
        "topography": case["topography"] if case["topography"] in TOPOGRAPHIES else UNKNOWN,
    }


def build_app(generator, cfg, device):
    meta = json.loads((DEMO_DIR / "cases.json").read_text(encoding="utf-8"))
    cases = {c["label"]: c for c in meta["cases"]}
    labels = list(cases)
    ranges = meta["ranges"]
    first = case_defaults(meta, cases[labels[0]])

    def generate(label, temp, hum, wind_speed, miss_t, miss_h, miss_w,
                 log_ha, wind_dir, topography, n_samples, seed):
        case = cases[label]
        ha = 10 ** log_ha
        cond, mask = build_condition(meta, case, (temp, hum, wind_speed), (miss_t, miss_h, miss_w),
                                     ha, wind_dir, topography)
        n = int(n_samples)
        gen = torch.Generator().manual_seed(int(seed))
        z = torch.randn(n, cfg.model.z_dim, generator=gen).to(device)
        cond_t = torch.from_numpy(cond).unsqueeze(0).expand(n, -1).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).expand(n, -1).to(device)
        veg = np.array(case["veg"], dtype=np.float32)
        veg_t = torch.from_numpy(veg).unsqueeze(0).expand(n, -1).to(device)

        with torch.no_grad():
            imgs = generator(z, cond_t, mask_t, veg_t)
        imgs = ((imgs + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        fires = [fire_intensity(im) for im in imgs]
        errors = [direction_error_deg(f, wind_dir) for f in fires]

        unknown = [name for name, m in (("temperatura", miss_t), ("humedad", miss_h),
                                         ("viento", miss_w)) if m]
        unknown += [name for name, v in (("dirección", wind_dir), ("topografía", topography)) if v == UNKNOWN]
        summary = (f"**Escenario:** {label.split(' · ')[0]} · {f'{ha:,.0f}'.replace(',', '.')} ha · "
                   f"{'?' if miss_t else f'{temp:.0f}'} °C · {'?' if miss_h else f'{hum:.0f}'} % HR · "
                   f"{'?' if miss_w else f'{wind_speed:.0f}'} km/h · viento {wind_dir} · topografía {topography}")
        if unknown:
            summary += f"  \n**Datos desconocidos para el modelo:** {', '.join(unknown)}"
        return render_samples(fires, veg, wind_dir, errors), summary + "  \n" + confidence_text(wind_dir, errors)

    with gr.Blocks(title="IncendiosGAN — Escenarios de propagación") as demo:
        gr.Markdown("# IncendiosGAN — Escenarios de propagación de incendios forestales")
        gr.Markdown(
            "Elige un incendio histórico de CONAF y modifica viento, clima o superficie. Un cGAN condicional "
            "genera **mapas de calor probabilísticos** de propagación (64×64) sobre la vegetación real del "
            "caso. Cada muestra es una realización posible: la variación entre ellas es incertidumbre, no ruido."
        )
        with gr.Row():
            with gr.Column(scale=1):
                case_dd = gr.Dropdown(labels, value=labels[0], label="Incendio histórico")
                wind_dir = gr.Dropdown(WIND_DIRS + [UNKNOWN], value=first["wind_dir"],
                                       label="Dirección del viento (de dónde sopla)")
                log_ha = gr.Slider(*LOG_HA_RANGE, value=first["log_ha"], step=0.1,
                                   label="Superficie final — log10(ha)",
                                   info="0 = 1 ha · 2 = 100 ha · 4 = 10.000 ha")
                temp = gr.Slider(*ranges["temp"], value=first["temp"], step=1, label="Temperatura (°C)")
                miss_t = gr.Checkbox(value=first["miss_t"], label="Temperatura desconocida")
                hum = gr.Slider(*ranges["hum"], value=first["hum"], step=1, label="Humedad relativa (%)")
                miss_h = gr.Checkbox(value=first["miss_h"], label="Humedad desconocida")
                wind_speed = gr.Slider(*ranges["wind_speed"], value=first["wind_speed"], step=1,
                                       label="Velocidad del viento (km/h)")
                miss_w = gr.Checkbox(value=first["miss_w"], label="Velocidad del viento desconocida")
                topo = gr.Dropdown(TOPOGRAPHIES + [UNKNOWN], value=first["topography"], label="Topografía")
                n_samples = gr.Slider(1, 9, value=6, step=1, label="Número de muestras")
                seed = gr.Number(value=42, label="Semilla aleatoria", precision=0)
                btn = gr.Button("Generar escenarios", variant="primary")
            with gr.Column(scale=2):
                out_img = gr.Image(label="Escenarios generados", type="numpy", interactive=False)
                out_txt = gr.Markdown()

        inputs = [case_dd, temp, hum, wind_speed, miss_t, miss_h, miss_w, log_ha, wind_dir, topo,
                  n_samples, seed]
        outputs = [out_img, out_txt]
        controls = [temp, hum, wind_speed, miss_t, miss_h, miss_w, log_ha, wind_dir, topo]

        def load_case(label):
            d = case_defaults(meta, cases[label])
            return [d["temp"], d["hum"], d["wind_speed"], d["miss_t"], d["miss_h"], d["miss_w"],
                    d["log_ha"], d["wind_dir"], d["topography"]]

        case_dd.change(load_case, case_dd, controls).then(generate, inputs, outputs)
        btn.click(generate, inputs, outputs)
        demo.load(generate, inputs, outputs)

        gr.Markdown(
            "---\n"
            "*Modelo: Conditional DCGAN (run 48, época 70) entrenado con ~6.500 registros CONAF, usando "
            "imágenes de un autómata celular calibrado con cada registro y enmascaramiento de condiciones "
            "para tolerar datos incompletos. Es una herramienta de investigación (Memoria de Título, UTFSM), "
            "no una predicción operacional ni una imagen satelital. Con incendios pequeños el modelo "
            "produce manchas poco definidas.*"
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Demo Gradio IncendiosGAN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pesos del generador (default: demo/generator_run48.pt)")
    parser.add_argument("--share", action="store_true", help="Crear link público de Gradio")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(DEMO_DIR / "generator_run48.pt")
    print(f"Cargando generador desde: {ckpt_path} | Device: {device}")

    demo = build_app(load_generator(ckpt_path, cfg, device), cfg, device)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
