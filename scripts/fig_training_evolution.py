#!/usr/bin/env python3
"""Genera figura de evolución del entrenamiento — Run 48 (para defensa).

Usa outputs/metrics/training_log_run48.csv
Columnas: epoch, d_loss, g_loss, r1, d_real, d_fake,
          dir_acc_real, dir_acc_fake, centroid_loss

Uso:
    python scripts/fig_training_evolution.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

LOG  = "outputs/metrics/training_log_run48.csv"
OUT  = "docs/figures/fig_training_evolution_run48.png"

BG   = "#FAFAFA"
TXT  = "#1A1A2E"

df = pd.read_csv(LOG)

# Suavizado opcional (ventana 3) para reducir ruido
def smooth(s, w=3):
    return s.rolling(w, center=True, min_periods=1).mean()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
fig.patch.set_facecolor(BG)

# ── Panel 1: Pérdidas G y D ──────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)
ax.plot(df["epoch"], smooth(df["g_loss"]), color="#2ECC71", lw=2, label="G loss")
ax.plot(df["epoch"], smooth(df["d_loss"]), color="#E74C3C", lw=2, label="D loss")
ax.set_xlabel("Época", fontsize=10)
ax.set_ylabel("Pérdida", fontsize=10)
ax.set_title("Pérdidas G y D", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# ── Panel 2: Precisión del clasificador auxiliar (AC-GAN) ────────────────────
ax = axes[1]
ax.set_facecolor(BG)
ax.plot(df["epoch"], smooth(df["dir_acc_real"]) * 100,
        color="#3498DB", lw=2, label="Acc. real (D entrenado)")
ax.plot(df["epoch"], smooth(df["dir_acc_fake"]) * 100,
        color="#F39C12", lw=2, label="Acc. fake (G condicional)")
ax.axhline(87.5, color="#888", lw=1, ls="--", label="1/8 × 7 = 87.5% ref.")
ax.set_xlabel("Época", fontsize=10)
ax.set_ylabel("Precisión direccional (%)", fontsize=10)
ax.set_title("Clasificador auxiliar AC-GAN\n(8 direcciones de viento)", fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# ── Panel 3: Señal real/fake del discriminador ───────────────────────────────
ax = axes[2]
ax.set_facecolor(BG)
ax.plot(df["epoch"], smooth(df["d_real"]), color="#1ABC9C", lw=2, label="D(real)")
ax.plot(df["epoch"], smooth(df["d_fake"]), color="#9B59B6", lw=2, label="D(fake)")
ax.axhline(0, color="#888", lw=1, ls="--")
ax.set_xlabel("Época", fontsize=10)
ax.set_ylabel("Puntuación discriminador", fontsize=10)
ax.set_title("Señal real vs. falso\n(WGAN — escores no acotados)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# Época óptima (checkpoint best = época 70)
for ax in axes:
    ax.axvline(70, color="#E74C3C", lw=1.5, ls=":", alpha=0.7)

# Anotación de época 70 solo en panel 1
axes[0].annotate("Checkpoint\nóptimo\n(época 70)", xy=(70, axes[0].get_ylim()[1] * 0.85),
                 fontsize=8, color="#E74C3C", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#E74C3C"),
                 xytext=(70 + 5, axes[0].get_ylim()[1] * 0.70))

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Guardado: {OUT}")
