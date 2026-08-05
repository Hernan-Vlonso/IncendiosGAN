#!/usr/bin/env python3
"""Figura de motivación para la defensa.

Panel izquierdo: hectáreas quemadas por año (2011-2021) con spike 2017.
Panel derecho:  mapa de dispersión de incendios coloreado por tamaño.

Salida: docs/figures/fig_motivacion.png

Uso:
    python scripts/fig_motivacion.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

from src.config import get_config

OUT = Path("docs/figures/fig_motivacion.png")

BG  = "#FAFAFA"
TXT = "#1A1A2E"
NARANJA = "#D85A1A"
GRIS    = "#AAAAAA"


def main():
    cfg = get_config()
    df  = pd.read_csv(cfg.paths.processed_csv)
    df["Año"] = pd.to_datetime(df["Inicio"], errors="coerce").dt.year

    # Coordenadas válidas de Chile
    df_geo = df[
        (df["Lat Decimal"] > -56) & (df["Lat Decimal"] < -17) &
        (df["Lon Decimal"] > -76) & (df["Lon Decimal"] < -65) &
        (df["Sup_Total"]   > 0)
    ].copy()

    ha_año = df.groupby("Año")["Sup_Total"].sum()

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)

    # ── Panel 1: barras por año ────────────────────────────────────────────────
    años   = ha_año.index.tolist()
    vals   = ha_año.values / 1_000          # en miles de ha
    colores = [NARANJA if a == 2017 else "#5B9BD5" for a in años]

    bars = ax1.bar(años, vals, color=colores, width=0.7, edgecolor="white", linewidth=0.5)
    ax1.set_facecolor(BG)
    ax1.set_xlabel("Año", fontsize=11, color=TXT)
    ax1.set_ylabel("Miles de hectáreas quemadas", fontsize=11, color=TXT)
    ax1.set_title("Impacto anual — Región del Maule\n(Dataset CONAF 2011–2021)",
                  fontsize=11, fontweight="bold", color=TXT)
    ax1.tick_params(labelsize=10, colors=TXT)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(GRIS)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k ha"))
    ax1.set_xticks(años)
    ax1.set_xticklabels([str(a) for a in años], rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3, color=GRIS)

    # Etiqueta 2017 — texto a la izquierda de la barra, fuera de ella
    idx_2017 = años.index(2017)
    ax1.set_ylim(0, vals[idx_2017] * 1.18)   # espacio sobre la barra
    ax1.annotate(
        "245.374 ha (SIDCO)\n+8× media años norm.",
        xy=(2017, vals[idx_2017]),
        xytext=(2014.2, vals[idx_2017] * 1.05),
        fontsize=9, color=NARANJA, fontweight="bold",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=NARANJA, lw=1.6,
                        connectionstyle="arc3,rad=-0.25"),
    )

    # Línea de media (sin 2017)
    media = np.mean([v for a, v in zip(años, vals) if a != 2017])
    ax1.axhline(media, color=GRIS, lw=1.2, ls="--", alpha=0.8)
    ax1.text(años[-1] + 0.1, media + 0.3, f"Media\n{media:.0f}k ha",
             fontsize=8, color=GRIS, va="bottom")

    # ── Panel 2: mapa geográfico con basemap ──────────────────────────────────
    # Ordenar por tamaño para que los grandes queden encima
    df_plot = df_geo.sort_values("Sup_Total")

    # Reproyectar a Web Mercator para contextily
    gdf = gpd.GeoDataFrame(
        df_plot,
        geometry=gpd.points_from_xy(df_plot["Lon Decimal"], df_plot["Lat Decimal"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    scatter = ax2.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=df_plot["Sup_Total"].values,
        cmap="hot_r",
        norm=LogNorm(vmin=0.1, vmax=df_plot["Sup_Total"].max()),
        s=np.clip(df_plot["Sup_Total"] / 20, 2, 120),
        alpha=0.85,
        linewidths=0,
        zorder=3,
    )

    # Basemap CartoDB Positron — muestra costa, ciudades y territorio claramente
    ctx.add_basemap(ax2, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.Positron,
                    zoom=8, attribution=False, zorder=1)

    cb = plt.colorbar(scatter, ax=ax2, pad=0.01, shrink=0.75, aspect=25)
    cb.set_label("Sup. quemada (ha)", fontsize=8, color=TXT)
    cb.ax.tick_params(labelsize=7.5, colors=TXT)

    ax2.set_xlabel("", fontsize=10, color=TXT)
    ax2.set_ylabel("", fontsize=10, color=TXT)
    ax2.set_title(
        f"Distribución geográfica — {len(df_geo):,} incendios\n"
        "Región del Maule  |  Cada punto = un incendio",
        fontsize=11, fontweight="bold", color=TXT,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Marcar Las Máquinas
    maquinas = df_geo[df_geo["Nombre incendio"].str.upper().str.contains("MAQUINAS", na=False)]
    if not maquinas.empty:
        r = maquinas.iloc[0]
        gdf_maq = gpd.GeoDataFrame(
            [r], geometry=[Point(r["Lon Decimal"], r["Lat Decimal"])],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        mx, my = float(gdf_maq.geometry.x.iloc[0]), float(gdf_maq.geometry.y.iloc[0])
        ax2.scatter(mx, my, s=220, c=NARANJA, marker="*", zorder=5)
        ax2.annotate(
            "Las Máquinas\n(159.812 ha)",
            xy=(mx, my),
            xytext=(mx + 8000, my - 12000),
            fontsize=7.5, color=NARANJA, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NARANJA, lw=1.2),
            zorder=6,
        )

    plt.tight_layout(pad=2.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
