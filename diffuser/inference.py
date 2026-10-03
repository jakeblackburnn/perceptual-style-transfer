"""Single source of truth for running a trained diffusion model on an image.

Mirrors style_transfer/inference.py's shape: preprocess/postprocess reused
directly from there (both packages agree on plain [0, 1]-range image
tensors at the PIL boundary), select_device likewise reused rather than
re-derived.
"""

import torch
from PIL import Image

from style_transfer.inference import preprocess, postprocess, select_device  # noqa: F401

from .models import ConditionedUNet
from .schedule import build_sampling_scheduler, to_diffusion_space, from_diffusion_space


def load_model(model_path, cfg, device="cpu"):
    """Load a trained ConditionedUNet from a checkpoint file."""
    model = ConditionedUNet(cfg.unet, cfg.image_size).to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def sample(model, content_image01: torch.Tensor, cfg, device, generator=None) -> torch.Tensor:
    """Run the reverse diffusion process, conditioned on content_image01 ([0, 1] range).

    Uses scheduler.step(...), whose clip_sample=True config clamps the
    scheduler's internal predicted-x0 to [-1, 1] on every step -- unlike the
    exploratory notebook's hand-rolled sampling loop, which only clamped the
    final output and let intermediate x_t drift unbounded across the reverse
    process.
    """
    scheduler = build_sampling_scheduler(cfg)
    content_diff = to_diffusion_space(content_image01).to(device)
    x_t = torch.randn(content_diff.shape, device=device, generator=generator)

    for t in scheduler.timesteps:
        t_batch = t.unsqueeze(0).to(device).expand(x_t.shape[0])
        eps_pred = model(x_t, content_diff, t_batch)
        x_t = scheduler.step(eps_pred, t, x_t).prev_sample

    return from_diffusion_space(x_t)


def stylize_image(content_img: Image.Image, model: ConditionedUNet, cfg, device="cpu") -> Image.Image:
    """Run the trained diffusion model on a PIL image and return the stylized PIL image."""
    content_tensor = preprocess(content_img).unsqueeze(0).to(device)
    output_tensor = sample(model, content_tensor, cfg, device)
    return postprocess(output_tensor.squeeze(0).cpu())
