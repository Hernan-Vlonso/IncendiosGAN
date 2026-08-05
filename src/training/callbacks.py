"""Callbacks para el training loop: checkpoints, sampling, logging."""

import math
from pathlib import Path
from collections import defaultdict

import torch
import torchvision.utils as vutils

# Vectores unitarios downwind en coordenadas de imagen (y hacia abajo)
# Índice: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
_SQRT2 = math.sqrt(2) / 2.0
_DIR_TARGETS_CB = [
    ( 0.0,    +1.0   ),  # N
    (-_SQRT2, +_SQRT2),  # NE
    (-1.0,     0.0   ),  # E
    (-_SQRT2, -_SQRT2),  # SE
    ( 0.0,    -1.0   ),  # S
    (+_SQRT2, -_SQRT2),  # SW
    (+1.0,     0.0   ),  # W
    (+_SQRT2, +_SQRT2),  # NW
]


class Callback:
    """Clase base para callbacks."""
    def on_train_start(self, trainer): pass
    def on_epoch_end(self, trainer, metrics): pass
    def on_train_end(self, trainer): pass


class CheckpointCallback(Callback):
    """Guardar checkpoints periódicamente."""

    def __init__(self, save_dir: str, interval: int = 25):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval

    def on_epoch_end(self, trainer, metrics):
        if trainer.epoch % self.interval == 0:
            path = self.save_dir / f"checkpoint_epoch{trainer.epoch:04d}.pt"
            trainer.save_checkpoint(str(path))
            print(f"  Checkpoint guardado: {path.name}")

    def on_train_end(self, trainer):
        path = self.save_dir / "checkpoint_final.pt"
        trainer.save_checkpoint(str(path))
        print(f"  Checkpoint final guardado: {path.name}")


class SamplingCallback(Callback):
    """Generar y guardar samples periódicamente."""

    def __init__(self, save_dir: str, n_samples: int = 64,
                 interval: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_samples = n_samples
        self.interval = interval
        self.fixed_z = None
        self.fixed_cond = None
        self.fixed_mask = None
        self.fixed_veg = None

    def on_train_start(self, trainer):
        """Fijar z y condiciones para comparar evolución."""
        self.fixed_z = torch.randn(
            self.n_samples, trainer.cfg.model.z_dim,
            device=trainer.device,
        )
        # Obtener condiciones fijas del val_loader
        if trainer.val_loader is not None:
            batch = next(iter(trainer.val_loader))
            n = min(self.n_samples, len(batch["condition"]))
            self.fixed_cond = batch["condition"][:n].to(trainer.device)
            self.fixed_mask = batch["mask"][:n].to(trainer.device)
            self.fixed_veg = batch["veg_profile"][:n].to(trainer.device)
            # Ajustar z al mismo tamaño
            self.fixed_z = self.fixed_z[:n]

    def on_epoch_end(self, trainer, metrics):
        if trainer.epoch % self.interval != 0:
            return
        if self.fixed_cond is None:
            return

        # Generar con EMA
        trainer.ema.shadow.eval()
        with torch.no_grad():
            samples = trainer.ema.shadow(
                self.fixed_z, self.fixed_cond, self.fixed_mask, self.fixed_veg
            )

        # Desnormalizar [-1,1] → [0,1]
        samples = (samples + 1) / 2

        # Guardar grid
        grid = vutils.make_grid(samples, nrow=8, normalize=False, padding=2)
        path = self.save_dir / f"samples_epoch{trainer.epoch:04d}.png"
        vutils.save_image(grid, str(path))

        # También al tensorboard
        trainer.writer.add_image("samples/ema", grid, trainer.epoch)


class BestCheckpointCallback(Callback):
    """Guarda checkpoint_best.pt cuando la diversidad inter-muestra es máxima.

    Evita perder el mejor modelo aunque el entrenamiento colapse después.
    """

    def __init__(self, save_dir: str, n_samples: int = 32, interval: int = 5):
        self.save_dir = Path(save_dir)
        self.n_samples = n_samples
        self.interval = interval
        self.best_diversity = 0.0
        self.best_epoch = 0
        self._fixed_z = None
        self._fixed_cond = None
        self._fixed_mask = None
        self._fixed_veg = None

    def on_train_start(self, trainer):
        self._fixed_z = torch.randn(
            self.n_samples, trainer.cfg.model.z_dim, device=trainer.device
        )
        if trainer.val_loader is not None:
            batch = next(iter(trainer.val_loader))
            n = min(self.n_samples, len(batch["condition"]))
            self._fixed_cond = batch["condition"][:n].to(trainer.device)
            self._fixed_mask = batch["mask"][:n].to(trainer.device)
            self._fixed_veg = batch["veg_profile"][:n].to(trainer.device)
            self._fixed_z = self._fixed_z[:n]

    def on_epoch_end(self, trainer, metrics):
        if trainer.epoch % self.interval != 0:
            return
        if self._fixed_cond is None:
            return

        trainer.ema.shadow.eval()
        with torch.no_grad():
            samples = trainer.ema.shadow(
                self._fixed_z, self._fixed_cond, self._fixed_mask, self._fixed_veg
            )

        diversity = samples.std(dim=0).mean().item()

        if diversity > self.best_diversity:
            self.best_diversity = diversity
            self.best_epoch = trainer.epoch
            path = self.save_dir / "checkpoint_best.pt"
            trainer.save_checkpoint(str(path))
            print(f"  [Best] epoch={trainer.epoch} diversity={diversity:.4f} -> checkpoint_best.pt")


class BestDirectionalCallback(Callback):
    """Guarda checkpoint_best_dir.pt cuando mejora la alineación centroide-dirección.

    Proxy rápido: genera n_per_dir muestras EMA por cada dirección física (8)
    y mide v·t/(H/2) promedio. Positivo = centroide apunta en dirección correcta.
    Más directo que BestCheckpointCallback (que usa diversidad de píxeles).
    """

    def __init__(self, save_dir: str, n_per_dir: int = 4, interval: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_per_dir = n_per_dir
        self.interval = interval
        self.best_alignment = -float('inf')
        self.best_epoch = 0
        self._cond = self._mask = self._veg = self._z = self._targets = None

    def on_train_start(self, trainer):
        if trainer.val_loader is None:
            return

        # Recoger n_per_dir condiciones por dirección física del val set
        collected = defaultdict(list)
        for batch in trainer.val_loader:
            cond_b = batch['condition']
            mask_b = batch['mask']
            veg_b  = batch['veg_profile']
            wind   = cond_b[:, 14:22]
            for i in range(len(cond_b)):
                d = int(wind[i].argmax())
                if wind[i].sum() > 0.5 and d < 8 and len(collected[d]) < self.n_per_dir:
                    collected[d].append((cond_b[i], mask_b[i], veg_b[i]))
            if all(len(collected[d]) >= self.n_per_dir for d in range(8)):
                break

        all_cond, all_mask, all_veg, dir_idx = [], [], [], []
        for d in range(8):
            for cond, mask, veg in collected[d]:
                all_cond.append(cond); all_mask.append(mask)
                all_veg.append(veg);   dir_idx.append(d)

        if not all_cond:
            return

        self._cond    = torch.stack(all_cond).to(trainer.device)
        self._mask    = torch.stack(all_mask).to(trainer.device)
        self._veg     = torch.stack(all_veg).to(trainer.device)
        self._z       = torch.randn(len(all_cond), trainer.cfg.model.z_dim,
                                    device=trainer.device)
        self._targets = torch.tensor(
            [_DIR_TARGETS_CB[d] for d in dir_idx],
            dtype=torch.float32, device=trainer.device
        )

    def on_epoch_end(self, trainer, metrics):
        if trainer.epoch % self.interval != 0 or self._cond is None:
            return

        trainer.ema.shadow.eval()
        with torch.no_grad():
            fake = trainer.ema.shadow(self._z, self._cond, self._mask, self._veg)

        B, C, H, W = fake.shape
        fire  = (fake.mean(dim=1) + 1.0) / 2.0
        yy    = torch.arange(H, dtype=fake.dtype, device=fake.device) - (H - 1) / 2.0
        xx    = torch.arange(W, dtype=fake.dtype, device=fake.device) - (W - 1) / 2.0
        mass  = fire.sum(dim=(1, 2)).clamp(min=1e-6)
        cy    = (fire * yy.view(1, H, 1)).sum(dim=(1, 2)) / mass
        cx    = (fire * xx.view(1, 1, W)).sum(dim=(1, 2)) / mass

        v_dot_t   = (cx * self._targets[:, 0] + cy * self._targets[:, 1]) / (H / 2.0)
        alignment = v_dot_t.mean().item()

        if alignment > self.best_alignment:
            self.best_alignment = alignment
            self.best_epoch     = trainer.epoch
            path = self.save_dir / 'checkpoint_best_dir.pt'
            trainer.save_checkpoint(str(path))
            print(f'  [BestDir] epoch={trainer.epoch} alignment={alignment:.4f} -> checkpoint_best_dir.pt')


class MetricsLogger(Callback):
    """Logging de métricas a archivo CSV."""

    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

    def on_epoch_end(self, trainer, metrics):
        if not self._header_written:
            with open(self.save_path, "w") as f:
                f.write("epoch," + ",".join(metrics.keys()) + "\n")
            self._header_written = True

        with open(self.save_path, "a") as f:
            vals = ",".join(f"{v:.6f}" for v in metrics.values())
            f.write(f"{trainer.epoch},{vals}\n")
