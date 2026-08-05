#!/usr/bin/env python3
"""Script 08: Generar mapa georreferenciado de propagación de incendio.

Toma un incendio real del dataset (por índice o nombre), genera heatmaps
con la GAN usando sus condiciones reales, y los superpone sobre un mapa
de la región del Maule como imagen estática y/o mapa interactivo HTML.

Uso:
    python scripts/08_geomap_scenario.py --checkpoint PATH --idx 0
    python scripts/08_geomap_scenario.py --checkpoint PATH --name "LAS BUITRERAS"
    python scripts/08_geomap_scenario.py --checkpoint PATH --idx 0 --incomplete
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow
import contextily as ctx
from PIL import Image
import io

from src.config import get_config
from src.models.generator import Generator

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DIR_VIENTO_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Calma", "Missing"]
EXPOSICION_CATS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "Plano", "Missing"]
TOPOGRAFIA_CATS = ["Suave", "Irregular", "Abrupta", "Missing"]

# Tamaño real del grid CA en metros (estimación basada en área típica del dataset)
# El grid 64x64 representa el área de propagación del incendio
# Para un incendio de X has, la celda tiene sqrt(X*10000/64^2) metros de lado
GRID_CELLS = 64


def area_to_cell_size_m(sup_total_has: float) -> float:
    """Convierte superficie en Has a tamaño de celda en metros."""
    if sup_total_has <= 0:
        sup_total_has = 1.0
    area_m2 = sup_total_has * 10_000
    cell_m = np.sqrt(area_m2 / (GRID_CELLS * GRID_CELLS))
    # Clampear entre 5m y 200m para evitar extremos
    return float(np.clip(cell_m, 5.0, 200.0))


def latlon_to_extent(lat: float, lon: float, grid_size_m: float):
    """Calcula el bounding box del grid centrado en (lat, lon).

    Returns (west, east, south, north) en grados decimales.
    """
    # 1 grado lat ≈ 111,320 m; 1 grado lon ≈ 111,320 * cos(lat) m
    half_m = grid_size_m * GRID_CELLS / 2
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * np.cos(np.radians(lat)))
    return lon - dlon, lon + dlon, lat - dlat, lat + dlat


def build_condition_vector(row: pd.Series, scaler, incomplete: bool = False,
                           mask_wind_dir: bool = False):
    """Construye el condition vector y mask a partir de una fila del dataset.

    Args:
        incomplete:    Solo temperatura y humedad conocidas; resto faltante.
        mask_wind_dir: Oculta únicamente Dir_viento (→ Missing); resto sin cambios.
                       Útil para comparación limpia "con/sin dirección de viento".
    """
    cfg = get_config()

    condition = np.zeros(32, dtype=np.float32)
    mask = np.zeros(32, dtype=np.float32)

    # Valores continuos: Temperatura, Humedad_Relativa, Vel_Viento, Sup_Total
    cont_cols = ["Temperatura", "Humedad_Relativa", "Vel_Viento", "Sup_Total"]
    cont_vals = []
    for col in cont_cols:
        val = row.get(col, None)
        if pd.isna(val):
            cont_vals.append(None)
        else:
            cont_vals.append(float(val))

    # Aplicar log1p a Sup_Total (índice 3) antes del scaler,
    # igual que en src/data/preprocessing.py:normalize_continuous
    raw = [v if v is not None else 0.0 for v in cont_vals]
    raw[3] = float(np.log1p(raw[3]))  # log1p(Sup_Total)
    cont_array = np.array(raw, dtype=np.float32).reshape(1, -1)
    try:
        normalized = scaler.transform(cont_array)[0]
    except Exception:
        normalized = cont_array[0]

    for i, (val, norm) in enumerate(zip(cont_vals, normalized)):
        if val is not None and not incomplete:
            condition[i] = norm
        elif val is not None and incomplete and i < 2:
            # En modo incompleto: solo temperatura y humedad conocidas
            condition[i] = norm
        else:
            mask[28 + i] = 1.0  # miss_flag

    # Exposicion (dims 4-13)
    exp = str(row.get("Exposicion", "Missing"))
    if exp in EXPOSICION_CATS and not incomplete:
        condition[4 + EXPOSICION_CATS.index(exp)] = 1.0
    else:
        condition[4 + EXPOSICION_CATS.index("Missing")] = 1.0

    # Dir_viento (dims 14-23)
    # Se oculta si: modo incompleto, o mask_wind_dir activado
    dv = str(row.get("Dir_viento", "Missing"))
    if dv in DIR_VIENTO_CATS and not incomplete and not mask_wind_dir:
        condition[14 + DIR_VIENTO_CATS.index(dv)] = 1.0
    else:
        condition[14 + DIR_VIENTO_CATS.index("Missing")] = 1.0

    # Topografia (dims 24-27)
    topo = str(row.get("Topografia", "Missing"))
    if topo in TOPOGRAFIA_CATS and not incomplete:
        condition[24 + TOPOGRAFIA_CATS.index(topo)] = 1.0
    else:
        condition[24 + TOPOGRAFIA_CATS.index("Missing")] = 1.0

    return condition, mask


def generate_heatmaps(generator, condition, mask, veg_profile, n_samples=6,
                      device="cpu", seed=42):
    """Genera n_samples heatmaps con la GAN para las condiciones dadas."""
    cfg = get_config()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cond_t = torch.tensor(condition, dtype=torch.float32).unsqueeze(0).to(device)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
    veg_t = torch.tensor(veg_profile, dtype=torch.float32).unsqueeze(0).to(device)

    images = []
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, cfg.model.z_dim, device=device)
            img = generator(z, cond_t, mask_t, veg_t)
            # Denormalizar de [-1,1] a [0,1]
            img = (img.squeeze().cpu().numpy() + 1) / 2
            img = np.transpose(img, (1, 2, 0))  # CHW → HWC
            images.append(img)

    return images


def heatmap_to_alpha(img_rgb: np.ndarray) -> np.ndarray:
    """Convierte imagen RGB [0,1] a RGBA con transparencia en zonas de fondo."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    # Señal de fuego: diferencia frente al fondo verde/gris.
    warm = np.clip((r - g * 0.5 - b * 0.2) * 3.5, 0, 1)
    bright = np.clip((r + g + b) / 3 - 0.20, 0, 1) * 2.0
    fire_intensity = np.clip(np.maximum(warm, bright), 0, 1)
    # Gamma más alto → más contraste; alpha mínimo del 0.15 para el fondo difuso
    alpha = np.clip(np.power(fire_intensity, 0.35) * 1.2, 0, 1)
    rgba = np.dstack([img_rgb, alpha])
    return rgba


def wind_arrow_params(wind_dir: str):
    """Devuelve (dx, dy) normalizado para la dirección de PROPAGACIÓN del fuego.
    Convención meteorológica: el viento viene DE wind_dir, el fuego se propaga
    en la dirección opuesta (downwind).
    """
    # Ángulo de propagación (downwind): opuesto al origen del viento.
    # E=0°, N=90° (CCW), en coordenadas geográficas.
    propagation_angles = {
        "N": -90,   # viento del Norte → fuego va al Sur
        "NE": -135, # viento del NE → fuego va al SW
        "E": 180,   # viento del Este → fuego va al Oeste
        "SE": 135,  # viento del SE → fuego va al NW
        "S": 90,    # viento del Sur → fuego va al Norte
        "SW": 45,   # viento del SW → fuego va al NE
        "W": 0,     # viento del Oeste → fuego va al Este
        "NW": -45,  # viento del NW → fuego va al SE
        "Calma": None, "Missing": None,
    }
    angle = propagation_angles.get(wind_dir, None)
    if angle is None:
        return None
    rad = np.radians(angle)
    return np.cos(rad), np.sin(rad)


def load_burn_perimeter(geojson_path: str, extent):
    """Carga perímetro real quemado desde GeoJSON y filtra por extent."""
    import json
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    west, east, south, north = extent
    with open(geojson_path) as f:
        gj = json.load(f)

    patches = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            # Solo incluir si intersecta con el extent
            if (max(lons) >= west and min(lons) <= east and
                    max(lats) >= south and min(lats) <= north):
                xy = list(zip(lons, lats))
                patches.append(MplPolygon(xy, closed=True))
        elif geom["type"] == "MultiPolygon":
            for ring in geom["coordinates"]:
                coords = ring[0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                if (max(lons) >= west and min(lons) <= east and
                        max(lats) >= south and min(lats) <= north):
                    xy = list(zip(lons, lats))
                    patches.append(MplPolygon(xy, closed=True))

    if not patches:
        return None
    return PatchCollection(patches, facecolor="none", edgecolor="cyan",
                           linewidth=0.8, alpha=0.9, zorder=8)


def make_static_figure(images, row, extent, fire_idx, output_path,
                       incomplete=False, mask_wind_dir=False, burn_geojson=None):
    """Genera figura estática: mapa base + heatmaps + metadatos."""
    west, east, south, north = extent
    n = len(images)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows + 1.5),
                             squeeze=False)

    nombre = row.get("Nombre incendio", f"Incendio #{fire_idx}")
    comuna = row.get("Comuna", "")
    sup = row.get("Sup_Total", 0)
    dv = row.get("Dir_viento", "?")
    vv = row.get("Vel_Viento", "?")
    temp = row.get("Temperatura", "?")
    lat = row.get("Lat Decimal", 0)
    lon = row.get("Lon Decimal", 0)
    if incomplete:
        mode = " (datos incompletos)"
    elif mask_wind_dir:
        mode = " (Dir_viento=Missing)"
    else:
        mode = ""

    fig.suptitle(
        f"{nombre} — {comuna}{mode}\n"
        f"Lat: {lat:.4f}°, Lon: {lon:.4f}° | "
        f"Superficie: {sup:.1f} ha | Viento: {dv} {vv} km/h | T°: {temp}°C",
        fontsize=12, fontweight="bold", y=0.98
    )

    arrow = wind_arrow_params(str(dv))

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.set_visible(False)
            continue

        # Mapa base satelital
        try:
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.Esri.WorldImagery,
                zoom="auto",
                attribution=False,
            )
        except Exception:
            ax.set_facecolor("#2d5a27")

        # Overlay del heatmap
        rgba = heatmap_to_alpha(images[i])
        ax.imshow(rgba, extent=[west, east, south, north],
                  origin="upper", aspect="auto", zorder=3, interpolation="bilinear")

        # Punto de ignición (centro del grid)
        ax.plot(lon, lat, "r*", markersize=10, zorder=5, label="Ignición")

        # Flecha de viento (cian, visible sobre fondo oscuro)
        if arrow is not None:
            dx, dy = arrow
            grid_deg = (east - west) * 0.22
            ax.annotate(
                "", xy=(lon + dx * grid_deg, lat + dy * grid_deg * 0.75),
                xytext=(lon, lat),
                arrowprops=dict(arrowstyle="-|>", color="cyan",
                                lw=3, mutation_scale=20),
                zorder=6
            )
            ax.text(lon + dx * grid_deg * 1.18, lat + dy * grid_deg * 0.88,
                    f"Prop.\n({dv})", color="cyan", fontsize=8,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65),
                    zorder=7)

        # Perímetro real quemado (si disponible)
        if burn_geojson is not None:
            import copy
            from matplotlib.collections import PatchCollection
            pc = load_burn_perimeter(burn_geojson, extent)
            if pc is not None:
                ax.add_collection(copy.copy(pc))
                if i == 0:
                    ax.plot([], [], color="cyan", lw=1.5, label="Área quemada real")
                    ax.legend(loc="lower right", fontsize=7,
                              facecolor="black", labelcolor="white")

        ax.set_xlabel("Longitud", fontsize=8)
        ax.set_ylabel("Latitud", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Realización {i+1}", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mapa estático guardado: {output_path}")

    # Figura auxiliar: thumbnails crudos 64×64 para verificar diversidad
    thumb_path = str(output_path).replace(".png", "_thumbnails.png")
    fig2, axes2 = plt.subplots(1, len(images), figsize=(2.5 * len(images), 2.8))
    if len(images) == 1:
        axes2 = [axes2]
    stds = [float(np.std(im)) for im in images]
    for i, (ax2, im) in enumerate(zip(axes2, images)):
        ax2.imshow(im, origin="upper")
        ax2.set_title(f"R{i+1}  σ={stds[i]:.3f}", fontsize=8)
        ax2.axis("off")
    fig2.suptitle("Imágenes crudas generadas (64×64) — verificación de diversidad",
                  fontsize=9)
    plt.tight_layout()
    plt.savefig(thumb_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Thumbnails guardados: {thumb_path}")
    print(f"  Diversidad (std): {[f'{s:.3f}' for s in stds]}")


def make_interactive_map(images, row, extent, fire_idx, output_path, incomplete=False):
    """Genera mapa interactivo HTML con folium."""
    import folium
    from folium import plugins

    lat = row.get("Lat Decimal", -35.5)
    lon = row.get("Lon Decimal", -71.5)
    west, east, south, north = extent
    nombre = row.get("Nombre incendio", f"Incendio #{fire_idx}")
    dv = row.get("Dir_viento", "?")
    sup = row.get("Sup_Total", 0)
    mode = " (datos incompletos)" if incomplete else ""

    m = folium.Map(location=[lat, lon], zoom_start=13,
                   tiles="Esri.WorldImagery")

    # Agregar cada realización como overlay
    for i, img_rgb in enumerate(images):
        rgba = heatmap_to_alpha(img_rgb)
        # Convertir a PNG en memoria
        rgba_uint8 = (rgba * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgba_uint8, mode="RGBA")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        import base64
        img_b64 = base64.b64encode(buf.read()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[south, west], [north, east]],
            opacity=0.75,
            name=f"Realización {i+1}",
            show=(i == 0),
        ).add_to(m)

    # Marker de ignición
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b>{nombre}{mode}</b><br>"
            f"Superficie: {sup:.1f} ha<br>"
            f"Viento: {dv}<br>"
            f"Lat: {lat:.4f}, Lon: {lon:.4f}",
            max_width=200
        ),
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    print(f"  Mapa interactivo guardado: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--idx", type=int, default=None,
                   help="Índice del incendio en el dataset")
    p.add_argument("--name", type=str, default=None,
                   help="Nombre del incendio (búsqueda parcial)")
    p.add_argument("--n-samples", type=int, default=6,
                   help="Número de realizaciones a generar")
    p.add_argument("--incomplete", action="store_true",
                   help="Usar solo temperatura y humedad (resto faltante)")
    p.add_argument("--mask-wind-dir", action="store_true",
                   help="Ocultar solo Dir_viento (→ Missing); resto igual. "
                        "Comparación limpia con/sin dirección de viento.")
    p.add_argument("--interactive", action="store_true",
                   help="Generar también mapa HTML interactivo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/figures/geomaps")
    p.add_argument("--burn-geojson", default=None,
                   help="GeoJSON con perímetro real quemado para comparación")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Cargar datos
    df = pd.read_csv(cfg.paths.processed_csv)
    import pickle
    with open(cfg.paths.scaler_path, "rb") as f:
        scaler = pickle.load(f)
    condition_vectors = np.load(cfg.paths.condition_vectors)
    veg_profiles = np.load(cfg.paths.vegetation_profiles)

    # Seleccionar incendio
    if args.name is not None:
        mask_name = df["Nombre incendio"].str.upper().str.contains(
            args.name.upper(), na=False)
        matches = df[mask_name]
        if len(matches) == 0:
            print(f"No se encontró incendio con nombre '{args.name}'")
            sys.exit(1)
        fire_idx = matches.index[0]
        print(f"Incendio encontrado: {matches.iloc[0]['Nombre incendio']} (idx={fire_idx})")
    elif args.idx is not None:
        fire_idx = args.idx
    else:
        # Seleccionar un incendio grande con viento conocido por defecto
        valid = df[
            (df["Lat Decimal"] > -38) & (df["Lat Decimal"] < -34) &
            (df["Lon Decimal"] > -73) & (df["Lon Decimal"] < -70) &
            (df["Sup_Total"] > 10) & (df["Dir_viento"] != "Missing")
        ]
        fire_idx = valid.index[0]
        print(f"Usando incendio por defecto: {df.loc[fire_idx, 'Nombre incendio']} (idx={fire_idx})")

    row = df.loc[fire_idx]
    print(f"\nIncendio: {row['Nombre incendio']} — {row['Comuna']}")
    print(f"  Lat: {row['Lat Decimal']:.4f}, Lon: {row['Lon Decimal']:.4f}")
    print(f"  Superficie: {row['Sup_Total']:.1f} ha")
    print(f"  Viento: {row['Dir_viento']} {row['Vel_Viento']} km/h")
    print(f"  Temperatura: {row['Temperatura']}°C  Humedad: {row['Humedad_Relativa']}%")

    # Cargar generador
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        veg_dim=cfg.model.veg_dim,
    ).to(device)
    state = ckpt.get("ema", ckpt.get("generator", ckpt))
    generator.load_state_dict(state)
    generator.eval()
    print(f"\nGenerador cargado (epoch {ckpt.get('epoch', '?')})")

    # Construir condición
    condition, mask = build_condition_vector(
        row, scaler,
        incomplete=args.incomplete,
        mask_wind_dir=args.mask_wind_dir,
    )
    # Usar veg_profile del dataset procesado (mismo índice)
    veg_profile = veg_profiles[fire_idx]

    # Generar imágenes
    print(f"\nGenerando {args.n_samples} realizaciones...")
    images = generate_heatmaps(
        generator, condition, mask, veg_profile,
        n_samples=args.n_samples, device=device, seed=args.seed
    )

    # Calcular extent geográfico
    cell_m = area_to_cell_size_m(float(row["Sup_Total"]))
    grid_size_m = cell_m * GRID_CELLS
    extent = latlon_to_extent(
        float(row["Lat Decimal"]), float(row["Lon Decimal"]), cell_m
    )
    print(f"  Celda CA: {cell_m:.1f} m | Grid: {grid_size_m:.0f} m × {grid_size_m:.0f} m")

    # Guardar figuras
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = str(row["Nombre incendio"]).replace(" ", "_").replace("/", "-")
    if args.incomplete:
        suffix = "_incomplete"
    elif args.mask_wind_dir:
        suffix = "_no_wind_dir"
    else:
        suffix = ""
    static_path = out_dir / f"geomap_{fire_idx}_{safe_name}{suffix}.png"
    make_static_figure(images, row, extent, fire_idx, static_path,
                       incomplete=args.incomplete,
                       mask_wind_dir=args.mask_wind_dir,
                       burn_geojson=args.burn_geojson)

    if args.interactive:
        html_path = out_dir / f"geomap_{fire_idx}_{safe_name}{suffix}.html"
        make_interactive_map(images, row, extent, fire_idx, html_path,
                             incomplete=args.incomplete)

    print("\nListo.")


if __name__ == "__main__":
    main()
