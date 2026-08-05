#!/usr/bin/env python3
"""Gráfico de barras: error angular y variabilidad direccional — Run 48.

Muestra dos métricas separadas por ser conceptualmente distintas:
  - angular_error_deg: sesgo sistemático (|ángulo_medido − esperado|), siempre ≥ 0
  - angle_std_deg: variabilidad natural entre realizaciones (std del ángulo medido)

No se mezclan en barras de error porque std puede ser > error sin contradicción:
un error bajo con std alto significa que el modelo acierta en promedio pero las
realizaciones individuales son dispersas (incertidumbre aleatoria del dominio).

Salida: docs/figures/fig_directional_error_bars.png

Uso:
    python scripts/fig_directional_error_bars.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

JSON_IN = Path("outputs/metrics/directional_centroid_metric_run48.json")
OUT     = Path("docs/figures/fig_directional_error_bars.png")

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def main():
    with open(JSON_IN) as f:
        data = json.load(f)

    errors = [data[d]["angular_error_deg"] for d in DIRECTIONS]
    stds   = [data[d]["angle_std_deg"]     for d in DIRECTIONS]
    mean_err = data["summary"]["mean_angular_error_deg"]

    fig, (ax_err, ax_std) = plt.subplots(1, 2, figsize=(11, 5), facecolor="white")
    fig.suptitle(
        "Validación direccional — Run 48  ·  8/8 dirs. dentro de 45°",
        fontsize=12, fontweight="bold", color="#222244"
    )

    y = np.arange(len(DIRECTIONS))

    # ── Panel izquierdo: error angular sistemático ──────────────────────────
    ax_err.set_facecolor("#F8F9FA")
    colors_err = ["#2ecc71" if e <= 15 else "#f39c12" if e <= 30 else "#e74c3c"
                  for e in errors]
    ax_err.barh(y, errors, color=colors_err, alpha=0.85, height=0.6)

    ax_err.axvline(45,       color="#c0392b", lw=2,   linestyle="--")
    ax_err.axvline(mean_err, color="#2c3e50", lw=1.5, linestyle=":")

    for i, e in enumerate(errors):
        ax_err.text(e + 0.5, i, f"{e:.0f}°",
                    va="center", ha="left", fontsize=9.5, color="#333333")

    ax_err.set_yticks(y)
    ax_err.set_yticklabels(DIRECTIONS, fontsize=12, fontweight="bold")
    ax_err.set_xlabel("Error angular sistemático (°)", fontsize=10)
    ax_err.set_xlim(0, 55)
    ax_err.set_title(f"Error angular\n(media {mean_err:.1f}°)", fontsize=10)
    ax_err.text(45.5, -0.7, "45°", color="#c0392b", fontsize=8)
    ax_err.text(mean_err + 0.3, 7.4, f"{mean_err:.1f}°", color="#2c3e50", fontsize=8)
    ax_err.grid(axis="x", alpha=0.35, color="gray")
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)

    # ── Panel derecho: variabilidad entre realizaciones (std) ────────────────
    ax_std.set_facecolor("#F8F9FA")
    colors_std = ["#3498db" if s <= 20 else "#9b59b6" for s in stds]
    ax_std.barh(y, stds, color=colors_std, alpha=0.75, height=0.6)

    ax_std.axvline(45, color="#c0392b", lw=2, linestyle="--")

    for i, s in enumerate(stds):
        ax_std.text(s + 0.5, i, f"{s:.0f}°",
                    va="center", ha="left", fontsize=9.5, color="#333333")

    ax_std.set_yticks(y)
    ax_std.set_yticklabels([], fontsize=12)
    ax_std.set_xlabel("Desviación estándar del ángulo medido (°)", fontsize=10)
    ax_std.set_xlim(0, 55)
    ax_std.set_title("Variabilidad entre realizaciones\n(std del ángulo medido)", fontsize=10)
    ax_std.text(45.5, -0.7, "45°", color="#c0392b", fontsize=8)
    ax_std.grid(axis="x", alpha=0.35, color="gray")
    ax_std.spines["top"].set_visible(False)
    ax_std.spines["right"].set_visible(False)

    # ── Leyenda compartida ───────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#2ecc71", alpha=0.85, label="Error ≤ 15°"),
        mpatches.Patch(color="#f39c12", alpha=0.85, label="15° < Error ≤ 30°"),
        mpatches.Patch(color="#e74c3c", alpha=0.85, label="Error > 30°"),
        mpatches.Patch(color="#3498db", alpha=0.75, label="Std ≤ 20°"),
        mpatches.Patch(color="#9b59b6", alpha=0.75, label="Std > 20°"),
        plt.Line2D([0], [0], color="#c0392b", lw=2, linestyle="--", label="Umbral 45°"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
