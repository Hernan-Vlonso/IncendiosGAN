#!/usr/bin/env python3
"""Diagramas de arquitectura cGAN — dos figuras separadas para defensa.

Genera:
  docs/figures/fig_architecture_generator.png
  docs/figures/fig_architecture_discriminator.png

Reglas de diseño:
  - Solo flechas horizontales o verticales (sin diagonales ni elbows)
  - Coordenadas en pulgadas, nunca normalizadas
  - shrinkA/B=5 para que puntas no entren en las cajas
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

BG = "#FAFAFA"
C  = dict(inp="#4A90D9", emb="#7B68EE", gen="#27AE60",
          dis="#E74C3C", los="#E67E22", img="#1ABC9C")


def make_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.set_facecolor(BG); ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, label, sub="", color=C["gen"], fs=11):
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.07", lw=1.5,
        edgecolor="white", facecolor=color, alpha=0.93, zorder=3))
    dy = 0.15 if sub else 0
    ax.text(cx, cy+dy, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(cx, cy-0.22, sub, ha="center", va="center",
                fontsize=fs-1.5, color="white", alpha=0.90, zorder=4)


def harr(ax, x0, x1, y, color="#AAAAAA", lw=1.8):
    """Flecha horizontal pura."""
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>",
        color=color, linewidth=lw, mutation_scale=11,
        shrinkA=5, shrinkB=5, zorder=5))


def varr(ax, x, y0, y1, color="#AAAAAA", lw=1.8):
    """Flecha vertical pura."""
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>",
        color=color, linewidth=lw, mutation_scale=11,
        shrinkA=5, shrinkB=5, zorder=5))


def path(ax, xs, ys, color="#AAAAAA", lw=1.8):
    """Segmento de línea sin punta."""
    ax.plot(xs, ys, color=color, lw=lw,
            solid_capstyle="round", zorder=5)


def larr(ax, pts, color="#AAAAAA", lw=1.8, shrinkA=5, shrinkB=5):
    """Flecha multi-segmento (L, Z…) como un solo artista — sin cortes ni cambios de color.
    pts: lista de (x, y); la punta queda en el último punto.
    Usar shrinkA=0/shrinkB=0 cuando los extremos ya están desplazados del borde de la caja.
    """
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
    ax.add_patch(FancyArrowPatch(
        path=MPath(pts, codes),
        arrowstyle="-|>", color=color, linewidth=lw,
        mutation_scale=11, shrinkA=shrinkA, shrinkB=shrinkB, zorder=5))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — GENERADOR G
# ══════════════════════════════════════════════════════════════════════════════
def make_generator():
    W, H = 16, 5.5
    fig, ax = make_fig(W, H)

    # Fondo
    ax.add_patch(FancyBboxPatch((0.2, 0.4), W-0.4, 4.8,
        boxstyle="round,pad=0.1", lw=1.3,
        edgecolor=C["gen"], facecolor=C["gen"], alpha=0.06, zorder=1))
    ax.text(W/2, 5.25, "GENERADOR  G",
            fontsize=14, fontweight="bold", color=C["gen"], ha="center")

    # ── Posiciones ────────────────────────────────────────────────────────────
    # Entradas  (x_center=1.5, ancho=2.0, alto=0.70)
    INP = [(1.5, 4.8, "Ruido  z",      "128-dim",  C["inp"]),
           (1.5, 3.7, "Condición  c",  "32-dim",   C["inp"]),
           (1.5, 2.6, "Máscara  m",    "missing",  C["inp"]),
           (1.5, 1.5, "Vegetación  v", "10-dim",   C["inp"])]
    for cx, cy, lbl, sub, col in INP:
        box(ax, cx, cy, 2.0, 0.70, lbl, sub, col)

    # Condition Embedding  (centro x=4.6, cubre c/m/v: y=1.5–3.7 → cy=2.6, h=2.6)
    box(ax, 4.6, 2.6, 2.0, 2.6, "Condition\nEmbedding", "MLP → 128-dim", C["emb"])

    # Project z  (misma x que Emb, por encima)
    box(ax, 4.6, 4.8, 2.0, 0.70, "Project  z", "Linear → 4×4", C["gen"])

    # CBN Blocks  (centro x=8.8, cy=3.3, h=3.0 → rango 1.8–4.8)
    box(ax, 8.8, 3.3, 2.2, 3.0, "CBN Blocks\n(× 3)", "Upsample\n4×4 → 32×32", C["gen"])
    ax.text(8.8, 2.0, "← cond. en cada bloque",
            fontsize=8.5, color="white", ha="center", style="italic",
            alpha=0.85, zorder=4)

    # ConvT + Tanh  (x=12.2, cy=3.3)
    box(ax, 12.2, 3.3, 2.0, 0.80, "ConvT + Tanh", "→ 3×64×64", C["gen"])

    # Imagen generada  (x=14.8, cy=3.3)
    box(ax, 14.8, 3.3, 1.9, 0.80, "Imagen\nGenerada", "64×64", C["img"])

    # ── Flechas (solo horizontal o vertical) ─────────────────────────────────

    # c, m, v → Emb (horizontal a misma altura)
    harr(ax, 2.5, 3.6, 3.7, C["inp"])   # c
    harr(ax, 2.5, 3.6, 2.6, C["inp"])   # m
    harr(ax, 2.5, 3.6, 1.5, C["inp"])   # v

    # z → Project (horizontal)
    harr(ax, 2.5, 3.6, 4.8, C["inp"])

    # Project → CBN: L-path — desde borde derecho de Project z hasta borde izquierdo de CBN
    larr(ax, [(5.6, 4.8), (6.1, 4.8), (6.1, 4.6), (7.7, 4.6)], C["gen"],
         shrinkA=0, shrinkB=0)

    # Emb → CBN (condicionamiento, horizontal a y=2.6)
    harr(ax, 5.6, 7.7, 2.6, C["emb"])

    # CBN → ConvT (horizontal)
    harr(ax, 9.9, 11.2, 3.3)

    # ConvT → Imagen (horizontal)
    harr(ax, 13.2, 13.85, 3.3)

    # ── Leyenda ───────────────────────────────────────────────────────────────
    items = [mpatches.Patch(color=C["inp"], label="Entradas"),
             mpatches.Patch(color=C["emb"], label="Condition Embedding"),
             mpatches.Patch(color=C["gen"], label="Bloques generador"),
             mpatches.Patch(color=C["img"], label="Imagen 3×64×64")]
    ax.legend(handles=items, loc="lower center",
              bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
              ncol=4, fontsize=9.5, framealpha=0.9, edgecolor="#CCCCCC")

    plt.savefig("docs/figures/fig_architecture_generator.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Guardado: fig_architecture_generator.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — DISCRIMINADOR D + PÉRDIDAS
# ══════════════════════════════════════════════════════════════════════════════
def make_discriminator():
    W, H = 16, 7.0
    fig, ax = make_fig(W, H)

    # Fondo D: desde y=1.9 hasta y=6.8
    ax.add_patch(FancyBboxPatch((0.2, 1.9), W-0.4, 4.9,
        boxstyle="round,pad=0.1", lw=1.3,
        edgecolor=C["dis"], facecolor=C["dis"], alpha=0.06, zorder=1))
    ax.text(W/2, H-0.28,
            "DISCRIMINADOR  D  +  Pérdidas de entrenamiento",
            fontsize=13, fontweight="bold", color=C["dis"], ha="center")

    # ── Cajas ─────────────────────────────────────────────────────────────────
    # Flujo principal (fila superior, y≈5.5–6.0)
    box(ax,  1.8, 6.0, 2.0, 0.75, "Imagen\nReal / Gen.", "3×64×64",              C["img"])
    box(ax,  5.2, 5.5, 2.2, 1.8,  "SN-Conv\n(× 4)",     "LeakyReLU\n64→512 ch", C["dis"])
    box(ax,  8.8, 5.5, 2.2, 0.75, "GlobalAvgPool",       "+ MiniBatchStd",       C["dis"])
    # Columna derecha: Proyección → CEmb → DirHead (apilados)
    box(ax, 13.0, 6.0, 3.0, 0.75, "Proyección + Linear", "→ score real/fake",    C["dis"])
    box(ax, 13.0, 4.5, 2.4, 0.75, "Cond. Embedding",     "compartida con G",     C["emb"])
    box(ax, 13.0, 3.1, 3.0, 0.75, "Dir. Head (AC-GAN)",  "→ 8 clases dir.",      C["dis"])

    # ── Flechas ───────────────────────────────────────────────────────────────

    # Imagen → SN-Conv  (y=6.0 cae dentro del rango SN-Conv 4.6–6.4)
    harr(ax, 2.8, 4.1, 6.0)

    # SN-Conv → GAP  (horizontal a y=5.5)
    harr(ax, 6.3, 7.7, 5.5)

    # GAP → Proyección y GAP → DirHead: desde borde derecho de GAP (9.9) hasta borde izquierdo (11.5)
    larr(ax, [(9.9, 5.5), (10.5, 5.5), (10.5, 6.0), (11.5, 6.0)],
         shrinkA=0, shrinkB=0)   # → Proyección
    larr(ax, [(9.9, 5.5), (10.5, 5.5), (10.5, 3.1), (11.5, 3.1)],
         shrinkA=0, shrinkB=0)   # → DirHead

    # CEmb → Proyección  (flecha vertical corta, sin cruzar nada)
    varr(ax, 13.0, 4.875, 5.625, C["emb"])

    # ── PÉRDIDAS ──────────────────────────────────────────────────────────────
    for lx, lbl, sub, bw in [
        (2.2,  "L_adv",      "Hinge + R1",        3.0),
        (5.8,  "L_AC",       "AC-GAN (dir.)",      3.0),
        (9.6,  "L_fm",       "Feature matching",   3.0),
        (13.8, "L_G = L_adv + L_AC + λ·L_fm",
               "L_D = L_adv + R1 + L_AC",          4.2),
    ]:
        box(ax, lx, 1.0, bw, 0.75, lbl, sub, C["los"], fs=10.5)

    # ── Leyenda ───────────────────────────────────────────────────────────────
    items = [
        mpatches.Patch(color=C["emb"], label="Cond. Embedding (compartido G y D)"),
        mpatches.Patch(color=C["dis"], label="Discriminador"),
        mpatches.Patch(color=C["los"], label="Pérdidas"),
        mpatches.Patch(color=C["img"], label="Imagen 3×64×64"),
    ]
    ax.legend(handles=items, loc="lower center",
              bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
              ncol=4, fontsize=9.5, framealpha=0.9, edgecolor="#CCCCCC")

    plt.savefig("docs/figures/fig_architecture_discriminator.png",
                dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Guardado: fig_architecture_discriminator.png")


if __name__ == "__main__":
    make_generator()
    make_discriminator()
