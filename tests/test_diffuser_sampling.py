"""Regression guard for the notebook's unclamped-intermediate-x_t failure mode.

The exploratory notebook's sampling loop only clamped the final output to
[0, 1] -- intermediate x_t was left free to drift across all 200 reverse
steps, and the sampled output ended up wildly out of range (dark,
red-blown-out, high-variance). diffuser's scheduler.step(..., clip_sample=
True) clamps the internal predicted-x0 every step; this test asserts that
guarantee holds end to end.
"""

import torch

from diffuser.config.schema import DiffusionExperimentConfig, SamplingConfig
from diffuser.config.unet_presets import get_unet_preset
from diffuser.models import ConditionedUNet
from diffuser.schedule import build_sampling_scheduler, to_diffusion_space, from_diffusion_space
from diffuser.inference import sample


def _tiny_cfg():
    return DiffusionExperimentConfig(
        image_size=8,
        unet=get_unet_preset("small"),
        num_train_timesteps=50,
        sampling=SamplingConfig(num_inference_steps=5),
    )


def test_intermediate_x_t_stays_bounded_during_sampling():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = ConditionedUNet(cfg.unet, cfg.image_size).eval()
    content01 = torch.rand(1, 3, cfg.image_size, cfg.image_size)

    scheduler = build_sampling_scheduler(cfg)
    content_diff = to_diffusion_space(content01)
    x_t = torch.randn(content_diff.shape, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        for t in scheduler.timesteps:
            t_batch = t.unsqueeze(0).expand(x_t.shape[0])
            eps_pred = model(x_t, content_diff, t_batch)
            x_t = scheduler.step(eps_pred, t, x_t).prev_sample
            # headroom above the [-1, 1] clamp for the variance-adding term
            assert x_t.min() >= -1.5, f"x_t drifted below bound: min={x_t.min().item()}"
            assert x_t.max() <= 1.5, f"x_t drifted above bound: max={x_t.max().item()}"


def test_sample_output_is_within_unit_range():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = ConditionedUNet(cfg.unet, cfg.image_size).eval()
    content01 = torch.rand(1, 3, cfg.image_size, cfg.image_size)

    output01 = sample(model, content01, cfg, device=torch.device("cpu"))

    assert output01.min() >= 0.0
    assert output01.max() <= 1.0
