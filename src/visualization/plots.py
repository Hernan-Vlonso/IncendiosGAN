"""Visualizaciones: loss curves, distribuciones, heatmaps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(history: dict, save_path: str = None):
    """Graficar curvas de loss durante entrenamiento.

    Args:
        history: dict con listas de d_loss, g_loss, r1, d_real, d_fake
        save_path: ruta para guardar la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["d_loss"]) + 1)

    # D loss
    axes[0, 0].plot(epochs, history["d_loss"], label="D Loss", color="tab:blue")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Discriminator Loss (Hinge)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # G loss
    axes[0, 1].plot(epochs, history["g_loss"], label="G Loss", color="tab:orange")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Generator Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # D scores (real vs fake)
    if "d_real" in history and "d_fake" in history:
        axes[1, 0].plot(epochs, history["d_real"], label="D(real)", color="tab:green")
        axes[1, 0].plot(epochs, history["d_fake"], label="D(fake)", color="tab:red")
        axes[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("Discriminator Scores")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    elif "w_dist" in history:
        axes[1, 0].plot(epochs, history["w_dist"], label="W Distance", color="tab:green")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Distance")
        axes[1, 0].set_title("Wasserstein Distance")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # R1 penalty
    r1_key = "r1" if "r1" in history else "gp"
    r1_label = "R1 Penalty" if "r1" in history else "Gradient Penalty"
    axes[1, 1].plot(epochs, history[r1_key], label=r1_label, color="tab:red")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Penalty")
    axes[1, 1].set_title(r1_label)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_condition_distribution(condition_vectors: np.ndarray,
                                col_names: list = None,
                                save_path: str = None):
    """Graficar distribuciones de las dimensiones del vector de condición."""
    n_dims = condition_vectors.shape[1]
    n_cont = 4  # primeras 4 son continuas

    fig, axes = plt.subplots(1, n_cont, figsize=(4 * n_cont, 4))
    if col_names is None:
        col_names = [f"Dim {i}" for i in range(n_dims)]

    for i in range(n_cont):
        axes[i].hist(condition_vectors[:, i], bins=50, alpha=0.7,
                      color="steelblue", edgecolor="white")
        axes[i].set_title(col_names[i])
        axes[i].set_xlabel("Valor")
        axes[i].set_ylabel("Frecuencia")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_missingness_heatmap(masks: np.ndarray, col_names: list = None,
                             save_path: str = None):
    """Heatmap de missingness por dimensión."""
    miss_rate = masks.mean(axis=0)

    fig, ax = plt.subplots(figsize=(14, 3))
    if col_names is None:
        col_names = [f"D{i}" for i in range(masks.shape[1])]

    sns.heatmap(
        miss_rate.reshape(1, -1),
        annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=col_names, yticklabels=["Miss rate"],
        ax=ax, cbar_kws={"label": "Missing rate"},
    )
    ax.set_title("Tasa de datos faltantes por dimensión")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_robustness_curves(masking_levels: list, fid_curve: list,
                           ssim_curve: list, save_path: str = None):
    """Curvas de FID y SSIM vs porcentaje de masking."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pcts = [p * 100 for p in masking_levels]

    # FID
    if fid_curve and fid_curve[0] is not None:
        ax1.plot(pcts, fid_curve, "o-", color="tab:red", linewidth=2, markersize=8)
        ax1.set_xlabel("Masking (%)")
        ax1.set_ylabel("FID")
        ax1.set_title("FID vs Datos Faltantes")
        ax1.grid(True, alpha=0.3)

    # SSIM
    ax2.plot(pcts, ssim_curve, "s-", color="tab:blue", linewidth=2, markersize=8)
    ax2.set_xlabel("Masking (%)")
    ax2.set_ylabel("SSIM")
    ax2.set_title("SSIM vs Datos Faltantes")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_ablation_robustness(masking_levels: list,
                             fid_masking: list, ssim_masking: list,
                             fid_baseline: list, ssim_baseline: list,
                             save_path: str = None):
    """Curvas comparativas de robustez: modelo con masking vs baseline.

    Args:
        masking_levels: lista de porcentajes de masking (e.g. [0.0, 0.1, ...])
        fid_masking: FID curve del modelo con condition masking
        ssim_masking: SSIM curve del modelo con condition masking
        fid_baseline: FID curve del modelo baseline (sin masking)
        ssim_baseline: SSIM curve del modelo baseline (sin masking)
        save_path: ruta para guardar la figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    pcts = [p * 100 for p in masking_levels]

    # FID
    ax1.plot(pcts, fid_masking, "o-", color="tab:blue", linewidth=2,
             markersize=8, label="Con Masking (mask_prob=0.3)")
    ax1.plot(pcts, fid_baseline, "s--", color="tab:red", linewidth=2,
             markersize=8, label="Baseline (mask_prob=0.0)")
    ax1.set_xlabel("Datos Faltantes (%)", fontsize=12)
    ax1.set_ylabel("FID", fontsize=12)
    ax1.set_title("FID vs Datos Faltantes — Ablation Study", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # SSIM
    ax2.plot(pcts, ssim_masking, "o-", color="tab:blue", linewidth=2,
             markersize=8, label="Con Masking (mask_prob=0.3)")
    ax2.plot(pcts, ssim_baseline, "s--", color="tab:red", linewidth=2,
             markersize=8, label="Baseline (mask_prob=0.0)")
    ax2.set_xlabel("Datos Faltantes (%)", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("SSIM vs Datos Faltantes — Ablation Study", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_vegetation_profiles(profiles: np.ndarray, veg_names: list = None,
                             n_samples: int = 20, save_path: str = None):
    """Visualizar perfiles de vegetación como barras apiladas."""
    if veg_names is None:
        veg_names = [f"Veg {i}" for i in range(profiles.shape[1])]

    fig, ax = plt.subplots(figsize=(14, 6))
    idx = np.random.default_rng(42).choice(len(profiles), size=n_samples, replace=False)
    subset = profiles[idx]

    bottom = np.zeros(n_samples)
    colors = plt.cm.Set3(np.linspace(0, 1, len(veg_names)))

    for i, (name, color) in enumerate(zip(veg_names, colors)):
        ax.bar(range(n_samples), subset[:, i], bottom=bottom,
               label=name, color=color, edgecolor="white", linewidth=0.5)
        bottom += subset[:, i]

    ax.set_xlabel("Registro (muestra)")
    ax.set_ylabel("Proporción")
    ax.set_title("Perfiles de Vegetación")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig
