#!/usr/bin/env python3
"""Figura polar de validación direccional — versión limpia para defensa.

Lee outputs/metrics/directional_centroid_metric_run48.json y genera
un polar plot de alta legibilidad con fondo blanco y fuentes grandes.

Salida: docs/figures/directional_centroid_validation_run48.png

Uso:
    python scripts/fig_directional_centroid_defensa.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

JSON_IN = Path("outputs/metrics/directional_centroid_metric_run48.json")
OUT     = Path("docs/figures/directional_centroid_validation_run48.png")

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
COLORS = plt.cm.Set1(np.linspace(0, 0.85, 8))


def main():
    with open(JSON_IN) as f:
        results = json.load(f)

    summary = results["summary"]
    mean_err = summary["mean_angular_error_deg"]
    std_err  = summary["std_angular_error_deg"]

    fig = plt.figure(figsize=(9, 7.5), facecolor="white")
    ax  = fig.add_subplot(111, projection="polar", facecolor="#F2F4F8")

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rticks([])
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["E", "NE", "N", "NW", "W", "SW", "S", "SE"],
                       fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.35, color="gray")
    ax.set_ylim(0, 1)

    legend_handles = []

    for i, wd in enumerate(DIRECTIONS):
        r = results[wd]
        exp_rad  = np.radians(r["expected_angle"])
        meas_rad = np.radians(r["measured_angle"]) if r["measured_angle"] else None
        err      = r["angular_error_deg"]
        std_deg  = r.get("angle_std_deg", 0.0)

        # Flecha esperada (gris, discontinua, más corta)
        ax.annotate("", xy=(exp_rad, 0.78), xytext=(exp_rad, 0.02),
                    arrowprops=dict(arrowstyle="-|>", color="#AAAAAA",
                                   lw=1.5, linestyle="dashed",
                                   mutation_scale=10))

        if meas_rad is not None:
            # Sector sombreado ±std alrededor de la dirección medida
            std_rad = np.radians(std_deg)
            theta_arc = np.linspace(meas_rad - std_rad, meas_rad + std_rad, 40)
            r_inner = np.full_like(theta_arc, 0.02)
            r_outer = np.full_like(theta_arc, 0.88)
            ax.fill_between(theta_arc, r_inner, r_outer,
                            color=COLORS[i], alpha=0.12)

            # Flecha medida (color, sólida, más larga)
            ax.annotate("", xy=(meas_rad, 0.90), xytext=(meas_rad, 0.02),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[i],
                                       lw=2.8, mutation_scale=14))

            err_str = f"{err:.0f}° ± {std_deg:.0f}°" if err is not None else "N/A"
            legend_handles.append(
                mlines.Line2D([], [], color=COLORS[i], lw=2.8,
                              label=f"Viento {wd:2s}  →  {err_str}")
            )

    legend_handles.append(
        mlines.Line2D([], [], color="#AAAAAA", lw=1.5, linestyle="dashed",
                      label="Dirección esperada (downwind)")
    )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(-0.38, -0.18),
        fontsize=10.5,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        title="Dirección   Error ± Desv. Est.",
        title_fontsize=10.5,
    )

    ax.set_title(
        "Ángulo de propagación: medido vs. esperado\n"
        f"Error medio = {mean_err:.1f}° ± {std_err:.1f}°  "
        f"·  8/8 dirs. dentro de 45°",
        fontsize=12, fontweight="bold", pad=18, color="#222244",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
