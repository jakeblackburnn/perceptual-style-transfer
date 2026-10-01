import torch

from diffuser.config.schema import DiffusionExperimentConfig
from diffuser.schedule import build_train_scheduler, predict_x0


def test_predict_x0_recovers_original_from_perfect_eps():
    cfg = DiffusionExperimentConfig(num_train_timesteps=1000)
    scheduler = build_train_scheduler(cfg)

    x0 = torch.rand(4, 3, 8, 8) * 2.0 - 1.0  # diffusion-space range [-1, 1]
    t = torch.tensor([0, 100, 500, 999])
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, t)

    # a "perfect" model just predicts the actual noise used
    x0_pred = predict_x0(scheduler, x_t, t, noise)

    # looser tolerance at high t: alpha_bar_t is near 0 there, so dividing by
    # its sqrt legitimately amplifies float32 rounding error regardless of
    # implementation correctness
    assert torch.allclose(x0_pred, x0, atol=1e-2)


def test_alphas_cumprod_monotonically_decreasing():
    cfg = DiffusionExperimentConfig(num_train_timesteps=1000)
    scheduler = build_train_scheduler(cfg)

    alphas_cumprod = scheduler.alphas_cumprod
    assert alphas_cumprod[0] > 0.99
    assert alphas_cumprod[-1] < 0.01
    assert torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1])


def test_default_beta_schedule_is_cosine():
    # guards against silently reverting to the notebook's linear schedule,
    # which wastes sampling steps at high noise (root cause of the failure)
    cfg = DiffusionExperimentConfig()
    assert cfg.beta_schedule == "squaredcos_cap_v2"

    scheduler = build_train_scheduler(cfg)
    assert scheduler.config.beta_schedule == "squaredcos_cap_v2"
