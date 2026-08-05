"""Pipeline de evaluación completa."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config, get_config
from src.evaluation.fid import FIDCalculator
from src.evaluation.ssim import compute_condition_matched_ssim
from src.evaluation.incomplete_data import evaluate_robustness


class Evaluator:
    """Pipeline de evaluación para la GAN condicional.

    Métricas:
    1. FID: Frechet Inception Distance
    2. SSIM: Structural Similarity condition-matched
    3. Robustez: curvas FID/SSIM vs porcentaje de masking
    """

    def __init__(self, cfg: Config = None, device: torch.device = None,
                 run_suffix: str = ""):
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fid_calc = FIDCalculator(self.device, cfg.eval.fid_batch_size)
        self.results = {}
        self._run_suffix = run_suffix  # e.g. "_run10" para no sobrescribir

    def prepare_real_images(self, dataloader, n_samples: int = None):
        """Cargar imágenes reales del dataloader y cachear stats FID.

        Args:
            dataloader: val dataloader
            n_samples: número máximo de imágenes

        Returns:
            real_images: tensor (N, 3, H, W) en [0, 1]
            real_conditions: array (N, 32)
            real_masks: array (N, 32)
            real_veg: array (N, veg_dim) o None
        """
        if n_samples is None:
            n_samples = self.cfg.eval.fid_n_samples

        images = []
        conditions = []
        masks = []
        vegs = []
        total = 0

        for batch in tqdm(dataloader, desc="Cargando imágenes reales"):
            imgs = (batch["image"] + 1) / 2  # [-1,1] → [0,1]
            images.append(imgs)
            conditions.append(batch["condition"].numpy())
            masks.append(batch["mask"].numpy())
            if "veg_profile" in batch:
                vegs.append(batch["veg_profile"].numpy())
            total += len(imgs)
            if total >= n_samples:
                break

        real_images = torch.cat(images, dim=0)[:n_samples]
        real_conditions = np.concatenate(conditions, axis=0)[:n_samples]
        real_masks = np.concatenate(masks, axis=0)[:n_samples]
        real_veg = np.concatenate(vegs, axis=0)[:n_samples] if vegs else None

        # Cachear stats de FID
        self.fid_calc.cache_real_stats(real_images)

        return real_images, real_conditions, real_masks, real_veg

    @torch.no_grad()
    def generate_images(self, generator, conditions: np.ndarray,
                        masks: np.ndarray, veg_profiles: np.ndarray = None,
                        n_samples: int = None):
        """Generar imágenes condicionadas.

        Returns:
            tensor (N, 3, H, W) en [0, 1]
        """
        if n_samples is None:
            n_samples = len(conditions)
        n = min(n_samples, len(conditions))

        generator.eval()
        z_dim = self.cfg.model.z_dim
        batch_size = self.cfg.eval.fid_batch_size
        generated = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            z = torch.randn(end - start, z_dim, device=self.device)
            cond = torch.from_numpy(conditions[start:end]).to(self.device)
            mask = torch.from_numpy(masks[start:end]).to(self.device)
            if veg_profiles is not None and self.cfg.model.veg_dim > 0:
                veg = torch.from_numpy(veg_profiles[start:end]).to(self.device)
            else:
                veg = torch.zeros(end - start, 0, device=self.device)
            imgs = generator(z, cond, mask, veg)
            imgs = (imgs + 1) / 2
            generated.append(imgs.cpu())

        return torch.cat(generated, dim=0)

    def evaluate_all(self, generator, val_loader, conditions, masks,
                     veg_profiles=None):
        """Ejecutar evaluación completa.

        Args:
            generator: modelo generador
            val_loader: dataloader de validación
            conditions: array (N, 32) todos los condition vectors (solo para robustez)
            masks: array (N, 32) todas las máscaras (solo para robustez)

        Returns:
            dict con todas las métricas
        """
        print("=" * 60)
        print("EVALUACIÓN COMPLETA")
        print("=" * 60)

        # 1. Preparar imágenes reales y extraer sus condiciones/máscaras del val set
        print("\n1. Preparando imágenes reales...")
        real_images, real_conditions, real_masks, real_veg = self.prepare_real_images(val_loader)

        # 2. Generar imágenes usando las condiciones del val set (alineación correcta)
        print("\n2. Generando imágenes...")
        gen_images = self.generate_images(generator, real_conditions,
                                          real_masks, real_veg)

        # 3. FID
        print("\n3. Calculando FID...")
        fid = self.fid_calc.compute_fid(gen_images)
        print(f"   FID = {fid:.2f}")

        # 4. SSIM condition-matched
        print("\n4. Calculando SSIM condition-matched...")
        ssim_result = compute_condition_matched_ssim(
            real_images, gen_images,
            real_conditions, real_conditions,
            k=self.cfg.eval.ssim_k_neighbors,
            n_samples=self.cfg.eval.ssim_n_samples,
        )
        print(f"   SSIM = {ssim_result['ssim_mean']:.4f} ± {ssim_result['ssim_std']:.4f}")

        # 5. Robustez
        print("\n5. Evaluando robustez ante datos incompletos...")
        robustness = evaluate_robustness(
            generator, real_images, real_conditions, real_masks,
            self.cfg, self.fid_calc,
            n_samples=self.cfg.eval.robustness_n_samples,
            device=self.device,
            veg_profiles=real_veg,
        )

        # Compilar resultados
        self.results = {
            "fid": fid,
            "ssim_mean": ssim_result["ssim_mean"],
            "ssim_std": ssim_result["ssim_std"],
            "robustness": robustness,
        }

        # Guardar (nombre por sufijo para no sobrescribir entre runs)
        filename = f"evaluation_results{self._run_suffix}.json"
        save_path = self.cfg.paths.metrics / filename
        with open(save_path, "w") as f:
            # Convertir numpy a listas para JSON
            serializable = {
                "fid": fid,
                "ssim_mean": ssim_result["ssim_mean"],
                "ssim_std": ssim_result["ssim_std"],
                "robustness": {
                    "masking_levels": robustness["masking_levels"],
                    "fid_curve": [
                        float(x) if x is not None else None
                        for x in robustness["fid_curve"]
                    ],
                    "ssim_curve": [float(x) for x in robustness["ssim_curve"]],
                },
            }
            json.dump(serializable, f, indent=2)
        print(f"\nResultados guardados en {save_path}")

        return self.results
