"""Funciones de pérdida para GAN: Hinge loss + R1 regularization."""

import torch
import torch.nn as nn
import torch.autograd as autograd


# ─── Hinge Loss (usado en BigGAN, StyleGAN2) ───────────────────────────

def hinge_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss para el generator: -E[D(G(z))]."""
    return -fake_scores.mean()


def hinge_discriminator_loss(real_scores: torch.Tensor,
                             fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss para el discriminator.

    D quiere: D(real) > 1 y D(fake) < -1.
    Una vez que los scores pasan esos umbrales, D deja de recibir gradiente,
    lo que naturalmente previene que D se vuelva demasiado fuerte.
    """
    real_loss = torch.nn.functional.relu(1.0 - real_scores).mean()
    fake_loss = torch.nn.functional.relu(1.0 + fake_scores).mean()
    return real_loss + fake_loss


# ─── Feature Matching Loss (Salimans et al. 2016) ──────────────────────

def feature_matching_loss(real_features: list, fake_features: list) -> torch.Tensor:
    """Feature matching loss: L1 entre activaciones medias de D para reales y falsos.

    Fuerza al generador a producir imágenes cuyas estadísticas internas
    (en todos los niveles de D) sean similares a las reales, evitando que
    colapse en texturas de fondo ignorando los patrones de fuego.

    real_features y fake_features son listas de tensores (B, C, H, W).
    Los reales deben llegar detacheados (sin gradiente).
    """
    loss = torch.tensor(0.0, device=real_features[0].device)
    for rf, ff in zip(real_features, fake_features):
        loss = loss + nn.functional.l1_loss(ff.mean(0), rf.mean(0).detach())
    return loss / len(real_features)


# ─── R1 Gradient Penalty (Mescheder et al. 2018) ───────────────────────

def r1_gradient_penalty(discriminator, real_imgs: torch.Tensor,
                        condition: torch.Tensor, mask: torch.Tensor,
                        gamma: float = 10.0,
                        veg_profile: torch.Tensor = None) -> torch.Tensor:
    """R1 gradient penalty — solo sobre datos reales.

    Más simple y estable que WGAN-GP. Penaliza gradientes grandes
    del discriminador solo en la distribución real.

    R1 = (gamma/2) * E[||∇D(x)||²]

    Ref: Mescheder et al. "Which Training Methods for GANs do actually
    Converge?" (ICML 2018). Usado en StyleGAN2.
    """
    real_imgs = real_imgs.detach().requires_grad_(True)

    if veg_profile is not None:
        real_scores = discriminator(real_imgs, condition, mask, veg_profile)
    else:
        real_scores = discriminator(real_imgs, condition, mask)

    gradients = autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_imgs,
        create_graph=True,
    )[0]

    # ||∇D(x)||² por sample
    grad_norm_sq = gradients.view(gradients.size(0), -1).pow(2).sum(dim=1)

    return (gamma / 2.0) * grad_norm_sq.mean()


# ─── Legacy (mantener por compatibilidad) ──────────────────────────────

def wasserstein_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def wasserstein_discriminator_loss(real_scores: torch.Tensor,
                                   fake_scores: torch.Tensor) -> torch.Tensor:
    return fake_scores.mean() - real_scores.mean()


def gradient_penalty(discriminator, real_imgs, fake_imgs,
                     condition, mask, lambda_gp=10.0):
    batch_size = real_imgs.size(0)
    device = real_imgs.device
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolated = discriminator(interpolated, condition, mask)
    gradients = autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
