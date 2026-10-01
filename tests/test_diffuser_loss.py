import torch

from style_transfer.feature_extractors.vgg import initialize_vgg
from style_transfer.loss import perceptual_loss
from diffuser.loss import diffusion_perceptual_loss

initialize_vgg(layer_preset="standard", device="cpu")


def _random_batch(n=2, size=32):
    # VGG19 needs at least ~32px input to survive its 5 max-pool stages
    # without collapsing to a 0x0 feature map.
    return torch.rand(n, 3, size, size)


def test_all_timesteps_above_cutoff_returns_zero_and_is_backwardable():
    x0_pred = _random_batch()
    x0_pred.requires_grad_(True)
    content = _random_batch()
    style = _random_batch(n=1)
    timesteps = torch.tensor([500, 600])

    loss = diffusion_perceptual_loss(x0_pred, content, style, timesteps, cutoff_timestep=200)

    assert loss.item() == 0.0
    loss.backward()  # must not raise, even though nothing was masked in


def test_all_timesteps_below_cutoff_matches_direct_perceptual_loss():
    x0_pred = _random_batch()
    content = _random_batch()
    style = _random_batch(n=1)
    timesteps = torch.tensor([10, 50])

    gated = diffusion_perceptual_loss(x0_pred, content, style, timesteps, cutoff_timestep=200)
    direct = perceptual_loss(x0_pred, content, style)

    assert torch.allclose(gated, direct)


def test_mixed_timesteps_scores_only_masked_subset():
    x0_pred = _random_batch(n=3)
    content = _random_batch(n=3)
    style = _random_batch(n=1)
    timesteps = torch.tensor([10, 500, 20])  # only indices 0 and 2 qualify

    gated = diffusion_perceptual_loss(x0_pred, content, style, timesteps, cutoff_timestep=200)
    direct_on_subset = perceptual_loss(x0_pred[[0, 2]], content[[0, 2]], style)

    assert torch.allclose(gated, direct_on_subset)
