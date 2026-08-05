"""Training loop para Conditional DCGAN con Hinge loss + R1 regularization."""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import math

from src.config import Config
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.losses import (
    hinge_generator_loss, hinge_discriminator_loss,
    r1_gradient_penalty, feature_matching_loss,
)
from src.models.weights import init_weights
from src.models.diffaugment import DiffAugment


_SQRT2 = math.sqrt(2) / 2.0
# Expected downwind unit vectors in image coordinates (y-axis DOWN, x-axis RIGHT)
# Wind FROM direction N means fire spreads toward S = +y in image coords
# Indexed: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
_DIR_TARGETS = [
    ( 0.0,     +1.0),     # N   → fire goes S  (right/down)
    (-_SQRT2,  +_SQRT2),  # NE  → fire goes SW
    (-1.0,      0.0),     # E   → fire goes W
    (-_SQRT2,  -_SQRT2),  # SE  → fire goes NW
    ( 0.0,     -1.0),     # S   → fire goes N
    (+_SQRT2,  -_SQRT2),  # SW  → fire goes NE
    (+1.0,      0.0),     # W   → fire goes E
    (+_SQRT2,  +_SQRT2),  # NW  → fire goes SE
]


def centroid_direction_loss(fake_imgs: torch.Tensor,
                            condition: torch.Tensor) -> torch.Tensor:
    """Centroid direction loss para la GAN condicional de incendios.

    Formulacion Run 39: L = -(v . t) / (H/2)

    donde v = desplazamiento del centroide desde el centro de la imagen,
    t = vector unitario downwind esperado.

    Gradiente: d(L)/d(v) = -t/(H/2) — constante, siempre apunta hacia t.
    Sin loophole: fire al centro -> v=0 -> loss=0 pero gradiente != 0,
    empuja el fuego en direccion correcta desde el primer paso.

    La perdida se aplica solo a samples con direccion de viento conocida.
    """
    B, C, H, W = fake_imgs.shape

    targets = fake_imgs.new_tensor(_DIR_TARGETS)  # (8, 2)

    wind_onehot = condition[:, 14:22]        # (B, 8)
    valid = wind_onehot.sum(dim=1) > 0.5     # (B,) bool

    if not valid.any():
        return fake_imgs.new_zeros(()).requires_grad_(False)

    # Fire map: promedio de canales, desnormalizar [-1,1] -> [0,1]
    fire = (fake_imgs.mean(dim=1) + 1.0) / 2.0  # (B, H, W)

    # Coordenadas centradas
    yy = (torch.arange(H, dtype=fake_imgs.dtype, device=fake_imgs.device)
          - (H - 1) / 2.0)
    xx = (torch.arange(W, dtype=fake_imgs.dtype, device=fake_imgs.device)
          - (W - 1) / 2.0)

    mass = fire.sum(dim=(1, 2)).clamp(min=1e-6)
    cy = (fire * yy.view(1, H, 1)).sum(dim=(1, 2)) / mass  # (B,)
    cx = (fire * xx.view(1, 1, W)).sum(dim=(1, 2)) / mass  # (B,)

    v = torch.stack([cx, cy], dim=1)  # (B, 2)

    dir_idx = wind_onehot.argmax(dim=1)
    t = targets[dir_idx]                      # (B, 2)

    # Proyeccion escalar normalizada: positivo = centroide en direccion correcta
    v_dot_t = (v * t).sum(dim=1) / (H / 2.0)  # (B,)

    loss_per_sample = -v_dot_t                 # minimizar = maximizar alineacion
    return loss_per_sample[valid].mean()


class EMA:
    """Exponential Moving Average del generador."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)


class Trainer:
    """Entrenador para la Conditional DCGAN con Hinge + R1.

    Features:
    - Hinge loss: scores naturalmente acotados, D no se vuelve demasiado fuerte
    - R1 gradient penalty: solo sobre datos reales, más estable que WGAN-GP
    - EMA del generador para samples de mayor calidad
    - n_critic=1 (un paso D por cada paso G, estándar para hinge loss)
    """

    def __init__(self, cfg: Config, train_loader, val_loader=None,
                 callbacks=None):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu"
        )

        # Construir modelos
        self.generator = Generator(
            z_dim=cfg.model.z_dim,
            condition_dim=cfg.model.cond_input_dim // 2,
            cond_embed_dim=cfg.model.cond_embed_dim,
            cond_hidden_dim=cfg.model.cond_hidden_dim,
            img_channels=cfg.model.img_channels,
            features=cfg.model.g_features,
            veg_dim=cfg.model.veg_dim,
        ).to(self.device)

        self.discriminator = Discriminator(
            img_channels=cfg.model.img_channels,
            condition_dim=cfg.model.cond_input_dim // 2,
            cond_embed_dim=cfg.model.cond_embed_dim,
            cond_hidden_dim=cfg.model.cond_hidden_dim,
            features=cfg.model.d_features,
            veg_dim=cfg.model.veg_dim,
        ).to(self.device)

        # Inicializar pesos
        init_weights(self.generator)
        init_weights(self.discriminator)

        # Optimizadores
        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=cfg.training.lr_g,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.training.lr_d,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )

        # Schedulers — cosine annealing para estabilizar entrenamiento tardío
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, T_max=cfg.training.epochs, eta_min=cfg.training.lr_min
        )
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=cfg.training.epochs, eta_min=cfg.training.lr_min
        )

        # EMA
        self.ema = EMA(self.generator, decay=cfg.training.ema_decay)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(cfg.paths.metrics / "tensorboard"))

        # DiffAugment policy
        self.diffaug_policy = cfg.training.diffaug_policy

        # Estado
        self.epoch = 0
        self.global_step = 0
        self.history = {"d_loss": [], "g_loss": [], "r1": [], "d_real": [], "d_fake": [],
                        "dir_acc_real": [], "dir_acc_fake": [], "centroid_loss": []}

    def _train_discriminator(self, real_imgs, condition, mask, veg_profile):
        """Un paso de entrenamiento del discriminador con hinge loss + R1 + dir aux."""
        cfg = self.cfg.training
        batch_size = real_imgs.size(0)

        # Generar fakes (sin gradientes para G)
        z = torch.randn(batch_size, self.cfg.model.z_dim, device=self.device)
        with torch.no_grad():
            fake_imgs = self.generator(z, condition, mask, veg_profile)

        # Aplicar DiffAugment antes del discriminador
        real_aug = DiffAugment(real_imgs, self.diffaug_policy)
        fake_aug = DiffAugment(fake_imgs, self.diffaug_policy)

        # Scores + logits de dirección para reales Y fakes
        real_scores, real_dir_logits = self.discriminator(
            real_aug, condition, mask, veg_profile, return_dir=True
        )
        fake_scores, fake_dir_logits_d = self.discriminator(
            fake_aug, condition, mask, veg_profile, return_dir=True
        )

        # Hinge loss
        d_loss = hinge_discriminator_loss(real_scores, fake_scores)

        # Auxiliary direction loss en reales Y fakes (Run 45: AC-GAN completo)
        # Run 46: filtrar samples con Dir_viento enmascarado (masking bug fix)
        # Run 48: CE global (revertido desde per-direction de run47) + masking fix mantenido
        dir_acc_real = 0.0
        if cfg.lambda_dir_real > 0:
            dir_labels = condition[:, 14:24].argmax(dim=1)
            has_dir = condition[:, 14:22].sum(dim=1) > 0.5
            valid = (dir_labels < 8) & has_dir
            if valid.any():
                dir_loss_real   = F.cross_entropy(real_dir_logits[valid], dir_labels[valid])
                dir_loss_fake_d = F.cross_entropy(fake_dir_logits_d[valid], dir_labels[valid])
                d_loss = d_loss + cfg.lambda_dir_real * (dir_loss_real + dir_loss_fake_d)
                dir_acc_real = (
                    real_dir_logits[valid].argmax(1) == dir_labels[valid]
                ).float().mean().item()

        # R1 gradient penalty (cada r1_interval pasos para eficiencia)
        r1 = torch.tensor(0.0, device=self.device)
        if self.global_step % cfg.r1_interval == 0:
            r1 = r1_gradient_penalty(
                self.discriminator, real_imgs,
                condition, mask, cfg.r1_gamma, veg_profile,
            )

        total_loss = d_loss + r1

        self.opt_d.zero_grad()
        total_loss.backward()
        self.opt_d.step()

        return (d_loss.item(), r1.item(),
                real_scores.mean().item(), fake_scores.mean().item(), dir_acc_real)

    def _train_generator(self, batch_size, condition, mask, real_imgs=None, veg_profile=None):
        """Un paso de entrenamiento del generador con hinge loss + feature matching + dir aux."""
        cfg = self.cfg.training
        z = torch.randn(batch_size, self.cfg.model.z_dim, device=self.device)
        fake_imgs = self.generator(z, condition, mask, veg_profile)
        fake_aug = DiffAugment(fake_imgs, self.diffaug_policy)
        fake_scores, fake_dir_logits = self.discriminator(
            fake_aug, condition, mask, veg_profile, return_dir=True
        )

        g_loss = hinge_generator_loss(fake_scores)

        # Auxiliary direction loss: obliga a G a producir imágenes con señal direccional
        # Run 46: filtrar samples enmascarados (masking bug fix)
        # Run 48: CE global (revertido desde per-direction de run47) + masking fix mantenido
        dir_acc_fake = 0.0
        if cfg.lambda_dir_g > 0:
            dir_labels = condition[:, 14:24].argmax(dim=1)
            has_dir = condition[:, 14:22].sum(dim=1) > 0.5
            valid = (dir_labels < 8) & has_dir
            if valid.any():
                dir_loss_g = F.cross_entropy(fake_dir_logits[valid], dir_labels[valid])
                g_loss = g_loss + cfg.lambda_dir_g * dir_loss_g
                dir_acc_fake = (
                    fake_dir_logits[valid].argmax(1) == dir_labels[valid]
                ).float().mean().item()

        # Feature matching: fuerza a G a reproducir estadísticas de features de D
        fm_weight = self.cfg.training.fm_weight
        if fm_weight > 0 and real_imgs is not None:
            with torch.no_grad():
                real_feats = self.discriminator.extract_features(real_imgs)
            fake_feats = self.discriminator.extract_features(fake_imgs)
            fm_loss = feature_matching_loss(real_feats, fake_feats)
            g_loss = g_loss + fm_weight * fm_loss

        # Mode Seeking Loss (Mao et al. 2019): maximiza ||G(z1)-G(z2)|| / ||z1-z2||
        # A diferencia de pdist, tiene gradiente no-nulo incluso cuando G está colapsado,
        # porque z1 != z2 garantiza que los outputs difieren al menos en precisión numérica.
        diversity_weight = self.cfg.training.diversity_weight
        if diversity_weight > 0:
            z_alt = torch.randn(batch_size, self.cfg.model.z_dim, device=self.device)
            fake_alt = self.generator(z_alt, condition, mask, veg_profile).detach()

            # Normalizar por sqrt(dimensiones) para escala comparable al hinge loss
            img_dist = (fake_imgs - fake_alt).view(batch_size, -1).norm(dim=1) / (
                fake_imgs.size(1) * fake_imgs.size(2) * fake_imgs.size(3)
            ) ** 0.5
            z_dist = (z - z_alt).view(batch_size, -1).norm(dim=1).detach() / (
                self.cfg.model.z_dim ** 0.5
            )
            mode_seeking_loss = -((img_dist / (z_dist + 1e-5)).clamp(max=10.0)).mean()
            g_loss = g_loss + diversity_weight * mode_seeking_loss

        # Centroid direction loss (Run 34): empuja el centroide del fuego en la
        # dirección de propagación esperada para el viento de la condición.
        ctr_loss_val = 0.0
        if cfg.lambda_centroid > 0:
            ctr_loss = centroid_direction_loss(fake_imgs, condition)
            g_loss = g_loss + cfg.lambda_centroid * ctr_loss
            ctr_loss_val = ctr_loss.item()

        self.opt_g.zero_grad()
        g_loss.backward()
        # Gradient clipping
        if self.cfg.training.g_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.cfg.training.g_clip_norm
            )
        self.opt_g.step()

        # Actualizar EMA
        self.ema.update(self.generator)

        return g_loss.item(), ctr_loss_val, dir_acc_fake

    def train_epoch(self):
        """Entrenar una epoch completa."""
        self.generator.train()
        self.discriminator.train()

        cfg = self.cfg.training
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_r1 = 0
        epoch_d_real = 0
        epoch_d_fake = 0
        epoch_dir_acc_real = 0.0
        epoch_dir_acc_fake = 0.0
        epoch_centroid_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            real_imgs = batch["image"].to(self.device)
            condition = batch["condition"].to(self.device)
            mask = batch["mask"].to(self.device)
            veg_profile = batch["veg_profile"].to(self.device)

            # ── Train Discriminator ──
            d_loss, r1, d_real, d_fake, dir_acc = self._train_discriminator(
                real_imgs, condition, mask, veg_profile
            )
            epoch_d_loss += d_loss
            epoch_r1 += r1
            epoch_d_real += d_real
            epoch_d_fake += d_fake
            epoch_dir_acc_real += dir_acc

            # ── Train Generator (cada n_critic pasos de D) ──
            if (batch_idx + 1) % cfg.n_critic == 0:
                g_loss, ctr_loss, dir_acc_f = self._train_generator(
                    real_imgs.size(0), condition, mask, real_imgs, veg_profile
                )
                epoch_g_loss += g_loss
                epoch_centroid_loss += ctr_loss
                epoch_dir_acc_fake += dir_acc_f

            n_batches += 1
            self.global_step += 1

            # Logging en barra de progreso
            if (batch_idx + 1) % cfg.log_interval == 0:
                pbar.set_postfix({
                    "D": f"{d_loss:.3f}",
                    "G": f"{g_loss if 'g_loss' in dir() else 0:.3f}",
                    "Dr": f"{d_real:.2f}",
                    "Df": f"{d_fake:.2f}",
                })

        # Promedios
        n_g = max(n_batches // cfg.n_critic, 1)
        metrics = {
            "d_loss": epoch_d_loss / n_batches,
            "g_loss": epoch_g_loss / n_g,
            "r1": epoch_r1 / n_batches,
            "d_real": epoch_d_real / n_batches,
            "d_fake": epoch_d_fake / n_batches,
            "dir_acc_real": epoch_dir_acc_real / n_batches,
            "dir_acc_fake": epoch_dir_acc_fake / n_g,
            "centroid_loss": epoch_centroid_loss / n_g,
        }

        for key, val in metrics.items():
            self.history[key].append(val)
            self.writer.add_scalar(f"train/{key}", val, self.epoch)

        self.epoch += 1
        return metrics

    def train(self, n_epochs: int = None):
        """Entrenamiento completo."""
        if n_epochs is None:
            n_epochs = self.cfg.training.epochs

        print(f"Entrenando en {self.device} por {n_epochs} epochs")
        print(f"Loss: Hinge + R1 (gamma={self.cfg.training.r1_gamma})")
        print(f"Generator params: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(n_epochs):
            metrics = self.train_epoch()

            # Cosine annealing step
            self.sched_g.step()
            self.sched_d.step()

            lr_g = self.opt_g.param_groups[0]["lr"]
            print(f"Epoch {self.epoch}: D={metrics['d_loss']:.4f} "
                  f"G={metrics['g_loss']:.4f} "
                  f"D(real)={metrics['d_real']:.3f} "
                  f"D(fake)={metrics['d_fake']:.3f} "
                  f"Ctr={metrics['centroid_loss']:.4f} "
                  f"lr={lr_g:.2e}", flush=True)

            for cb in self.callbacks:
                cb.on_epoch_end(self, metrics)

        for cb in self.callbacks:
            cb.on_train_end(self)

        self.writer.close()
        print("Entrenamiento completado.")

    @torch.no_grad()
    def generate_samples(self, n_samples: int, conditions: torch.Tensor = None,
                         masks: torch.Tensor = None, veg_profiles: torch.Tensor = None,
                         use_ema: bool = True):
        """Generar imágenes con el generador (o EMA)."""
        gen = self.ema.shadow if use_ema else self.generator
        gen.eval()

        z = torch.randn(n_samples, self.cfg.model.z_dim, device=self.device)

        if conditions is None:
            batch = next(iter(self.val_loader))
            conditions = batch["condition"][:n_samples].to(self.device)
            masks = batch["mask"][:n_samples].to(self.device)
            veg_profiles = batch["veg_profile"][:n_samples].to(self.device)
        else:
            conditions = conditions.to(self.device)
            masks = masks.to(self.device)
            if veg_profiles is not None:
                veg_profiles = veg_profiles.to(self.device)

        return gen(z, conditions, masks, veg_profiles)

    def save_checkpoint(self, path: str):
        """Guardar checkpoint completo."""
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "ema": self.ema.shadow.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Cargar checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.generator.load_state_dict(ckpt["generator"])

        # Compatibilidad con checkpoints de runs anteriores que no tienen dir_head:
        # cargamos con strict=False y avisamos qué claves nuevas se inicializaron
        d_result = self.discriminator.load_state_dict(ckpt["discriminator"], strict=False)
        if d_result.missing_keys:
            print(f"  [compat] Discriminator: claves nuevas inicializadas aleatoriamente: "
                  f"{d_result.missing_keys}")
        if d_result.unexpected_keys:
            print(f"  [compat] Discriminator: claves ignoradas del checkpoint: "
                  f"{d_result.unexpected_keys}")

        self.ema.shadow.load_state_dict(ckpt["ema"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        try:
            self.opt_d.load_state_dict(ckpt["opt_d"])
        except ValueError:
            # El checkpoint es de antes de dir_head: el grupo de parámetros de opt_d
            # tiene tamaño distinto. Reiniciamos opt_d desde cero (solo se pierden
            # los momentum states de Adam, los pesos del discriminador ya están cargados).
            print("  [compat] opt_d reiniciado: checkpoint anterior a dir_head")
        if "sched_g" in ckpt:
            self.sched_g.load_state_dict(ckpt["sched_g"])
            self.sched_d.load_state_dict(ckpt["sched_d"])
        self.history = ckpt["history"]
        # Compatibilidad con checkpoints de runs anteriores
        if "dir_acc_real" not in self.history:
            self.history["dir_acc_real"] = []
        if "dir_acc_fake" not in self.history:
            self.history["dir_acc_fake"] = []
        if "centroid_loss" not in self.history:
            self.history["centroid_loss"] = []
        print(f"Checkpoint cargado (epoch {self.epoch})")
