#!/usr/bin/env python3
"""Script 06: Generar todas las figuras para la memoria de título.

Genera las figuras desde la terminal sin necesidad de Jupyter.
Requiere: outputs del entrenamiento y evaluación ya completados.

Uso: python scripts/06_generate_figures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")  # backend sin GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchvision.utils as vutils

from src.config import get_config
from src.models.generator import Generator
from src.data.dataset import FireDataset
from src.visualization.plots import (
    plot_training_curves, plot_robustness_curves,
    plot_condition_distribution, plot_missingness_heatmap,
    plot_vegetation_profiles,
)
from src.visualization.grid_viewer import (
    compare_real_vs_generated, show_image_grid, show_training_progression,
)
from src.visualization.condition_explorer import sweep_condition, sweep_masking


def main():
    cfg = get_config()
    fig_dir = cfg.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("GENERACIÓN DE FIGURAS PARA LA MEMORIA")
    print(f"Device: {device}")
    print(f"Output: {fig_dir}")
    print("=" * 60)

    # ---------- Cargar datos ----------
    cond = np.load(cfg.paths.condition_vectors)
    masks = np.load(cfg.paths.masks)
    veg = np.load(cfg.paths.vegetation_profiles)

    # ---------- Cargar resultados de evaluación ----------
    results_path = cfg.paths.metrics / "evaluation_results.json"
    with open(results_path) as f:
        results = json.load(f)
    rob = results["robustness"]

    # ---------- Cargar log de entrenamiento ----------
    log_path = cfg.paths.metrics / "training_log.csv"
    import pandas as pd
    log_df = pd.read_csv(log_path)
    history = {
        "d_loss": log_df["d_loss"].tolist(),
        "g_loss": log_df["g_loss"].tolist(),
        "r1": log_df["r1"].tolist(),
        "d_real": log_df["d_real"].tolist(),
        "d_fake": log_df["d_fake"].tolist(),
    }

    # ---------- Cargar generador ----------
    ckpt_path = cfg.paths.checkpoints / "checkpoint_final.pt"
    generator = Generator(
        z_dim=cfg.model.z_dim,
        condition_dim=cfg.model.cond_input_dim // 2,
        cond_embed_dim=cfg.model.cond_embed_dim,
        cond_hidden_dim=cfg.model.cond_hidden_dim,
        veg_dim=cfg.model.veg_dim,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt.get("ema", ckpt["generator"]))
    generator.eval()
    print(f"Generador cargado (epoch {ckpt.get('epoch', '?')})")

    # ---------- Cargar dataset de validación ----------
    val_dataset = FireDataset(
        str(cfg.paths.ca_images / "val"), cond, veg, masks,
        is_train=False, mask_prob=0.0,
    )

    # ==========================================================
    # FIGURA 1: Curvas de entrenamiento
    # ==========================================================
    print("\n[1/10] Curvas de entrenamiento...")
    plot_training_curves(history, save_path=str(fig_dir / "fig_training_curves.png"))

    # ==========================================================
    # FIGURA 2: Progresión del entrenamiento
    # ==========================================================
    print("[2/10] Progresión visual del entrenamiento...")
    show_training_progression(
        str(cfg.paths.samples), epochs=[5, 50, 100, 150, 200, 300],
        save_path=str(fig_dir / "fig_training_progression.png"),
    )

    # ==========================================================
    # FIGURA 3: Curvas de robustez
    # ==========================================================
    print("[3/10] Curvas de robustez...")
    plot_robustness_curves(
        rob["masking_levels"], rob["fid_curve"], rob["ssim_curve"],
        save_path=str(fig_dir / "fig_robustness_curves.png"),
    )

    # Versión anotada
    pcts = [p * 100 for p in rob["masking_levels"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(pcts, rob["fid_curve"], "o-", color="tab:red", linewidth=2.5, markersize=9)
    ax1.fill_between(pcts, rob["fid_curve"][0], rob["fid_curve"], alpha=0.1, color="tab:red")
    ax1.set_xlabel("Datos Faltantes (%)", fontsize=12)
    ax1.set_ylabel("FID", fontsize=12)
    ax1.set_title("FID vs Porcentaje de Datos Faltantes", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fid_0, fid_90 = rob["fid_curve"][0], rob["fid_curve"][-1]
    ax1.annotate(f"Δ = {((fid_90-fid_0)/fid_0)*100:+.1f}%",
                 xy=(90, fid_90), xytext=(60, fid_90 + 3),
                 fontsize=11, color="tab:red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red"))

    ax2.plot(pcts, rob["ssim_curve"], "s-", color="tab:blue", linewidth=2.5, markersize=9)
    ax2.fill_between(pcts, rob["ssim_curve"], rob["ssim_curve"][0], alpha=0.1, color="tab:blue")
    ax2.set_xlabel("Datos Faltantes (%)", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("SSIM vs Porcentaje de Datos Faltantes", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ssim_0, ssim_90 = rob["ssim_curve"][0], rob["ssim_curve"][-1]
    ax2.annotate(f"Δ = {((ssim_90-ssim_0)/ssim_0)*100:+.1f}%",
                 xy=(90, ssim_90), xytext=(60, ssim_90 - 0.02),
                 fontsize=11, color="tab:blue", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:blue"))
    plt.tight_layout()
    plt.savefig(str(fig_dir / "fig_robustness_annotated.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # FIGURA 4: Degradación relativa
    # ==========================================================
    print("[4/10] Degradación relativa...")
    fid_rel = [(f - fid_0) / fid_0 * 100 for f in rob["fid_curve"]]
    ssim_rel = [(s - ssim_0) / ssim_0 * 100 for s in rob["ssim_curve"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pcts, fid_rel, "o-", color="tab:red", linewidth=2, markersize=8, label="FID (↑ = peor)")
    ax.plot(pcts, ssim_rel, "s-", color="tab:blue", linewidth=2, markersize=8, label="SSIM (↓ = peor)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhspan(-10, 10, alpha=0.05, color="green", label="Zona ±10%")
    ax.set_xlabel("Datos Faltantes (%)", fontsize=12)
    ax.set_ylabel("Cambio Relativo (%)", fontsize=12)
    ax.set_title("Degradación Relativa de Métricas vs Datos Faltantes", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "fig_degradation_relative.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # FIGURA 5: Real vs Generado
    # ==========================================================
    print("[5/10] Comparación real vs generado...")
    n_compare = 8
    torch.manual_seed(42)
    real_imgs = torch.stack([val_dataset[i]["image"] for i in range(n_compare)])
    cond_batch = torch.stack([val_dataset[i]["condition"] for i in range(n_compare)]).to(device)
    mask_batch = torch.stack([val_dataset[i]["mask"] for i in range(n_compare)]).to(device)
    veg_batch = torch.stack([val_dataset[i]["veg_profile"] for i in range(n_compare)]).to(device)
    z = torch.randn(n_compare, cfg.model.z_dim, device=device)
    with torch.no_grad():
        gen_imgs = generator(z, cond_batch, mask_batch, veg_batch)
    compare_real_vs_generated(
        real_imgs, gen_imgs, n_show=n_compare,
        save_path=str(fig_dir / "fig_real_vs_generated.png"),
    )

    # ==========================================================
    # FIGURA 6: Grid de 64 samples
    # ==========================================================
    print("[6/10] Grid de 64 samples...")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    n_total = 64
    idx = rng.choice(len(cond), size=n_total, replace=False)
    z64 = torch.randn(n_total, cfg.model.z_dim, device=device)
    c64 = torch.from_numpy(cond[idx]).to(device)
    m64 = torch.from_numpy(masks[idx]).to(device)
    v64 = torch.from_numpy(veg[idx]).to(device)
    with torch.no_grad():
        gen_grid = generator(z64, c64, m64, v64)
    show_image_grid(
        gen_grid, nrow=8, title="64 Escenarios de Incendio Generados",
        save_path=str(fig_dir / "fig_generated_grid_64.png"),
    )

    # ==========================================================
    # FIGURA 7: Sweeps de condiciones
    # ==========================================================
    print("[7/10] Sweeps de condiciones...")
    base_cond = np.zeros(32, dtype=np.float32)
    base_cond[4] = 1.0   # Exposición: N
    base_cond[14] = 1.0  # Dir viento: N
    base_cond[25] = 1.0  # Topo: Irregular
    base_mask = np.zeros(32, dtype=np.float32)

    for dim_idx, dim_name, fname in [
        (0, "Temperatura (z-score)", "fig_sweep_temperatura.png"),
        (1, "Humedad (z-score)", "fig_sweep_humedad.png"),
        (2, "Vel. Viento (z-score)", "fig_sweep_viento.png"),
    ]:
        vals = np.linspace(-2, 2, 8)
        _, fig = sweep_condition(
            generator, base_cond, base_mask, dim_idx=dim_idx,
            values=vals, z_dim=cfg.model.z_dim, device=device,
            dim_name=dim_name,
        )
        fig.savefig(str(fig_dir / fname), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ==========================================================
    # FIGURA 8: Masking progresivo
    # ==========================================================
    print("[8/10] Masking progresivo...")
    full_cond = np.zeros(32, dtype=np.float32)
    full_cond[0] = 1.5
    full_cond[1] = -1.0
    full_cond[2] = 1.5
    full_cond[19] = 1.0  # SW
    full_cond[26] = 1.0  # Abrupta
    full_mask = np.zeros(32, dtype=np.float32)
    _, fig = sweep_masking(
        generator, full_cond, full_mask,
        dims_to_mask=[0, 1, 2, list(range(14, 24))],
        z_dim=cfg.model.z_dim, device=device,
    )
    fig.savefig(str(fig_dir / "fig_masking_progresivo.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ==========================================================
    # FIGURA 9: Diversidad (mismo z / misma cond)
    # ==========================================================
    print("[9/10] Figuras de diversidad...")
    torch.manual_seed(77)
    z_fixed = torch.randn(1, cfg.model.z_dim, device=device)
    idx_diverse = [0, 500, 1000, 2000, 4000]
    fig, axes = plt.subplots(1, len(idx_diverse), figsize=(3 * len(idx_diverse), 3.5))
    for i, idx_i in enumerate(idx_diverse):
        c_i = torch.from_numpy(cond[idx_i:idx_i+1]).to(device)
        m_i = torch.from_numpy(masks[idx_i:idx_i+1]).to(device)
        v_i = torch.from_numpy(veg[idx_i:idx_i+1]).to(device)
        with torch.no_grad():
            img = generator(z_fixed, c_i, m_i, v_i)
        img = (img + 1) / 2
        axes[i].imshow(img[0].cpu().permute(1, 2, 0).numpy())
        axes[i].set_title(f"Registro {idx_i}", fontsize=10)
        axes[i].axis("off")
    plt.suptitle("Mismo ruido z, diferentes condiciones", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(fig_dir / "fig_same_z_diff_cond.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Misma cond, diferente z
    c_fix = torch.from_numpy(cond[0:1]).expand(8, -1).to(device)
    m_fix = torch.from_numpy(masks[0:1]).expand(8, -1).to(device)
    v_fix = torch.from_numpy(veg[0:1]).expand(8, -1).to(device)
    torch.manual_seed(0)
    z_div = torch.randn(8, cfg.model.z_dim, device=device)
    with torch.no_grad():
        imgs_div = generator(z_div, c_fix, m_fix, v_fix)
    show_image_grid(
        imgs_div, nrow=8, title="Misma condición, 8 realizaciones (variando z)",
        save_path=str(fig_dir / "fig_same_cond_diff_z.png"),
    )

    # ==========================================================
    # FIGURA 10: Distribuciones y datos
    # ==========================================================
    print("[10/10] Distribuciones de datos...")
    plot_condition_distribution(
        cond, col_names=["Temperatura", "Humedad", "Vel_Viento", "Sup_Total"],
        save_path=str(fig_dir / "fig_condition_distributions.png"),
    )
    plot_vegetation_profiles(
        veg, veg_names=cfg.data.vegetation_cols, n_samples=30,
        save_path=str(fig_dir / "fig_vegetation_profiles.png"),
    )
    plot_missingness_heatmap(
        masks, save_path=str(fig_dir / "fig_missingness_heatmap.png"),
    )

    # ==========================================================
    # FIGURA RESUMEN COMPUESTA
    # ==========================================================
    print("\n[Bonus] Figura resumen compuesta...")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    epochs = range(1, len(history["d_loss"]) + 1)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(epochs, history["d_loss"], label="D Loss", color="tab:blue", alpha=0.7)
    ax_a.plot(epochs, history["g_loss"], label="G Loss", color="tab:orange", alpha=0.7)
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Loss")
    ax_a.set_title("(a) Curvas de Entrenamiento"); ax_a.legend(); ax_a.grid(True, alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(epochs, history["d_real"], label="D(real)", color="tab:green", alpha=0.7)
    ax_b.plot(epochs, history["d_fake"], label="D(fake)", color="tab:red", alpha=0.7)
    ax_b.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_b.set_xlabel("Epoch"); ax_b.set_ylabel("Score")
    ax_b.set_title("(b) Scores del Discriminador"); ax_b.legend(); ax_b.grid(True, alpha=0.3)

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(pcts, rob["fid_curve"], "o-", color="tab:red", linewidth=2, markersize=8)
    ax_c.set_xlabel("Datos Faltantes (%)"); ax_c.set_ylabel("FID")
    ax_c.set_title("(c) FID vs Datos Faltantes"); ax_c.grid(True, alpha=0.3)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(pcts, rob["ssim_curve"], "s-", color="tab:blue", linewidth=2, markersize=8)
    ax_d.set_xlabel("Datos Faltantes (%)"); ax_d.set_ylabel("SSIM")
    ax_d.set_title("(d) SSIM vs Datos Faltantes"); ax_d.grid(True, alpha=0.3)

    ax_e = fig.add_subplot(gs[2, :])
    torch.manual_seed(42)
    z16 = torch.randn(16, cfg.model.z_dim, device=device)
    idx16 = np.random.default_rng(42).choice(len(cond), size=16, replace=False)
    c16 = torch.from_numpy(cond[idx16]).to(device)
    m16 = torch.from_numpy(masks[idx16]).to(device)
    v16 = torch.from_numpy(veg[idx16]).to(device)
    with torch.no_grad():
        gen16 = generator(z16, c16, m16, v16)
    gen16 = (gen16 + 1) / 2
    grid_img = vutils.make_grid(gen16, nrow=8, padding=2).cpu().numpy().transpose(1, 2, 0)
    ax_e.imshow(grid_img)
    ax_e.set_title("(e) Muestras Generadas con Diferentes Condiciones")
    ax_e.axis("off")

    plt.savefig(str(fig_dir / "fig_resumen_resultados.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ==========================================================
    # RESUMEN
    # ==========================================================
    print("\n" + "=" * 60)
    print("FIGURAS GENERADAS")
    print("=" * 60)
    for f in sorted(fig_dir.glob("fig_*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} ({size_kb:.0f} KB)")
    print(f"\nTotal: {len(list(fig_dir.glob('fig_*.png')))} figuras en {fig_dir}")


if __name__ == "__main__":
    main()
