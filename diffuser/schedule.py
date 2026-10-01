"""Diffusion-space conversion and noise-schedule helpers.

style_transfer/ keeps all image tensors in plain [0, 1] range (see
style_transfer/inference.py's module docstring). Diffusion schedulers from
the `diffusers` library assume [-1, 1] range (their clip_sample_range=1.0
default clamps predicted x0 to that range). to_diffusion_space/
from_diffusion_space are the single boundary where that conversion happens,
so it can't drift out of sync the way the *255 mismatch CLAUDE.md documents
did for the feed-forward model.
"""

import torch
from diffusers import DDIMScheduler, DDPMScheduler


def to_diffusion_space(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def from_diffusion_space(x_pm1: torch.Tensor) -> torch.Tensor:
    return ((x_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)


def build_train_scheduler(cfg) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
    )


def build_sampling_scheduler(cfg) -> DDIMScheduler:
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
    )
    scheduler.set_timesteps(cfg.sampling.num_inference_steps)
    return scheduler


def predict_x0(scheduler, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
    """Recover predicted clean image (diffusion space, [-1, 1]) from eps_pred.

    Reads scheduler.alphas_cumprod -- the same public buffer diffusers'
    official training scripts use for SNR weighting -- rather than a
    hand-linspace'd schedule, so whichever beta_schedule the scheduler was
    built with (cosine by default here) flows through automatically.
    """
    alpha_bar_t = scheduler.alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
    x0 = (x_t - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
    return x0.clamp(-1.0, 1.0)
