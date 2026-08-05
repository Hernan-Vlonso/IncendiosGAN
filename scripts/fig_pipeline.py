#!/usr/bin/env python3
"""Pipeline del sistema — diagrama limpio para defensa."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "docs/figures/fig_pipeline_sistema.png"
BG  = "#FAFAFA"

C = {
    "data":  "#2980B9",
    "sim":   "#8E44AD",
    "train": "#27AE60",
    "infer": "#E67E22",
    "eval":  "#C0392B",
}

# ── Layout ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 6.2
BOX_W, BOX_H = 1.9, 1.05     # pulgadas por caja
GAP_H        = 0.70           # hueco vertical entre filas
MARGIN_X     = 0.55
GAP_X        = 0.28           # hueco horizontal entre cajas

# Posiciones: fila superior a 75%, inferior a 32% de FIG_H
Y_TOP = FIG_H * 0.75   # ~4.65" desde abajo
Y_BOT = FIG_H * 0.30   # ~1.86" desde abajo

def xs_row(n, margin=MARGIN_X):
    """Centros X de n cajas uniformemente distribuidas."""
    total = n * BOX_W + (n - 1) * GAP_X
    start = (FIG_W - total) / 2 + BOX_W / 2
    return [start + i * (BOX_W + GAP_X) for i in range(n)]

XS_TOP = xs_row(6)
XS_BOT = xs_row(5)

TOP_ITEMS = [
    ("Registros\nCONAF",           C["data"]),
    ("Extracción\ncondiciones",    C["data"]),
    ("Simulador CA\nMonte Carlo",  C["sim"]),
    ("Dataset\nsintético",         C["sim"]),
    ("Entrenamiento\ncGAN Run 48", C["train"]),
    ("Checkpoint\nG* guardado",    C["train"]),
]
BOT_ITEMS = [
    ("Nuevo incendio\ncondiciones",  C["data"]),
    ("Generador G*\ninferencia",     C["infer"]),
    ("Heatmaps\nsintéticos",         C["infer"]),
    ("Validación\ngeoespacial",      C["eval"]),
    ("Métricas\nfinales",            C["eval"]),
]

# ── Figura ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])   # ocupa toda la figura
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_facecolor(BG)
ax.axis("off")

def draw_box(cx, cy, label, color):
    rect = FancyBboxPatch(
        (cx - BOX_W/2, cy - BOX_H/2), BOX_W, BOX_H,
        boxstyle="round,pad=0.05", linewidth=1.4,
        edgecolor="white", facecolor=color, alpha=0.92, zorder=3,
        transform=ax.transData,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white",
            zorder=4, transform=ax.transData, linespacing=1.35)

def draw_arrow(x0, y0, x1, y1, color="#AAAAAA"):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        color=color, linewidth=1.6,
        mutation_scale=10,
        shrinkA=3, shrinkB=3,
        zorder=5,
        transform=ax.transData,
    )
    ax.add_patch(arr)

# ── Etiquetas de fase ─────────────────────────────────────────────────────────
kw = dict(ha="center", va="center", fontsize=9.5, fontweight="bold",
          color="#555", transform=ax.transData,
          bbox=dict(boxstyle="round,pad=0.2", fc="#EEEEEE", ec="#CCCCCC"))
ax.text(FIG_W/2, Y_TOP + BOX_H/2 + 0.38,
        "① FASE OFFLINE — Construcción del dataset y entrenamiento", **kw)
ax.text(FIG_W/2, Y_BOT + BOX_H/2 + 0.38,
        "② FASE ONLINE — Inferencia y validación", **kw)


# ── Cajas y flechas fila superior ─────────────────────────────────────────────
for cx, (label, color) in zip(XS_TOP, TOP_ITEMS):
    draw_box(cx, Y_TOP, label, color)

for i in range(len(XS_TOP)-1):
    draw_arrow(XS_TOP[i]+BOX_W/2, Y_TOP, XS_TOP[i+1]-BOX_W/2, Y_TOP)

# ── Cajas y flechas fila inferior ─────────────────────────────────────────────
for cx, (label, color) in zip(XS_BOT, BOT_ITEMS):
    draw_box(cx, Y_BOT, label, color)

for i in range(len(XS_BOT)-1):
    draw_arrow(XS_BOT[i]+BOX_W/2, Y_BOT, XS_BOT[i+1]-BOX_W/2, Y_BOT)

# ── Flecha codo: checkpoint → inferencia ─────────────────────────────────────
cx_ckpt = XS_TOP[5]
cx_inf  = XS_BOT[1]
y_turn  = (Y_TOP - BOX_H/2 + Y_BOT + BOX_H/2) / 2   # punto medio vertical

ax.plot([cx_ckpt, cx_ckpt], [Y_TOP - BOX_H/2, y_turn],
        color=C["train"], lw=1.8, zorder=5, solid_capstyle="butt",
        transform=ax.transData)
ax.plot([cx_ckpt, cx_inf],  [y_turn, y_turn],
        color=C["train"], lw=1.8, zorder=5, solid_capstyle="butt",
        transform=ax.transData)
arr = FancyArrowPatch(
    (cx_inf, y_turn), (cx_inf, Y_BOT + BOX_H/2),
    arrowstyle="-|>", color=C["train"], linewidth=1.8,
    mutation_scale=10, shrinkA=0, shrinkB=3, zorder=5,
    transform=ax.transData,
)
ax.add_patch(arr)
ax.text(cx_ckpt + 0.12, y_turn + 0.10,
        "pesos G*", fontsize=8.5, color=C["train"], fontweight="bold",
        ha="left", va="bottom", transform=ax.transData)

# ── Leyenda ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C["data"],  label="Datos / preprocesamiento"),
    mpatches.Patch(color=C["sim"],   label="Simulador CA Monte Carlo"),
    mpatches.Patch(color=C["train"], label="Entrenamiento cGAN"),
    mpatches.Patch(color=C["infer"], label="Inferencia"),
    mpatches.Patch(color=C["eval"],  label="Evaluación / validación"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure,
          ncol=5, fontsize=9, framealpha=0.9, edgecolor="#CCCCCC")

plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Guardado: {OUT}")
