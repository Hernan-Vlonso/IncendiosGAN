#!/usr/bin/env python3
"""Figura de robustez ante datos faltantes — versión limpia para defensa.

Lee outputs/metrics/evaluation_results_run48_ep70.json y genera
un gráfico de dos paneles con FID y SSIM vs % masking.

Salida: docs/figures/robustness_curves.png

Uso:
    python scripts/fig_robustness_defensa.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

JSON_IN = Path("outputs/metrics/evaluation_results_run48_ep70.json")
OUT     = Path("docs/figures/robustness_curves.png")
BG      = "#FAFAFA"
TXT     = "#1A1A2E"
AZUL    = "#2980B9"
VERDE   = "#27AE60"
ROJO    = "#E74C3C"


def main():
    d   = json.load(open(JSON_IN))
    rob = d["robustness"]
    x   = [m * 100 for m in rob["masking_levels"]]
    fid  = rob["fid_curve"]
    ssim = rob["ssim_curve"]

    # Layout vertical: FID arriba, SSIM abajo — ratio ~0.75:1 para llenar
    # la columna izquierda (0.44 textwidth) a altura similar a la imagen derecha
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7.5),
                                   gridspec_kw={"hspace": 0.45})
    fig.patch.set_facecolor(BG)

    # ── Panel 1: FID ──────────────────────────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.plot(x, fid, color=ROJO, lw=2.5, marker="o", ms=6, zorder=3)
    ax1.axvspan(0, 70, alpha=0.08, color=VERDE, label="Región robusta (≤70%)")
    ax1.axvline(70, color=VERDE, lw=1.5, ls="--", alpha=0.8)
    ax1.set_xlabel("Condiciones enmascaradas (%)", fontsize=10.5)
    ax1.set_ylabel("FID", fontsize=10.5)
    ax1.set_title("FID vs. datos faltantes", fontsize=11, fontweight="bold")
    ax1.set_ylim(270, 310)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9.5)
    ax1.spines[["top","right"]].set_visible(False)

    # ── Panel 2: SSIM ─────────────────────────────────────────────────────────
    ax2.set_facecolor(BG)
    ax2.plot(x, ssim, color=AZUL, lw=2.5, marker="o", ms=6, zorder=3)
    ax2.axhline(0.50, color="#AAAAAA", lw=1.2, ls=":", label="SSIM = 0.50")
    ax2.axvspan(0, 70, alpha=0.08, color=VERDE, label="Región robusta (≤70%)")
    ax2.axvline(70, color=VERDE, lw=1.5, ls="--", alpha=0.8)
    ax2.set_xlabel("Condiciones enmascaradas (%)", fontsize=10.5)
    ax2.set_ylabel("SSIM", fontsize=10.5)
    ax2.set_title("SSIM vs. datos faltantes", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.42, 0.56)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9.5)
    ax2.spines[["top","right"]].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUT), dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
