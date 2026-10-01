"""Perceptual-loss integration for diffusion training.

The exploratory notebook scored `perceptual_loss` against the model's
predicted-x0 at every sampled timestep, including near-pure-noise ones where
x0_pred is a poor estimate -- its large, poorly-conditioned gradient likely
fought the denoising MSE objective directly (the notebook's own training log
showed MSE trending up instead of down). diffusion_perceptual_loss restricts
scoring to low-noise timesteps only, per a configurable cutoff.
"""

import torch

from style_transfer.loss import perceptual_loss


def diffusion_perceptual_loss(x0_pred, content_images, style_images, timesteps, cutoff_timestep,
                               content_weight=1.0, style_weight=1e5):
    """Score x0_pred against content/style targets, restricted to low-noise timesteps.

    Only batch elements with timesteps[i] < cutoff_timestep contribute; the
    rest are masked out before scoring. style_images is passed through
    unmasked -- perceptual_loss already broadcasts across all content/style
    pairs independently of batch size, so it doesn't need to be indexed by
    the (unrelated) content-batch mask. Returns a zero-valued tensor if no
    batch element qualifies, still attached to x0_pred's graph (via a 0
    multiply) so callers can unconditionally call .backward() on the
    combined loss without a "does not require grad" error on high-noise
    training steps where nothing passes the cutoff.
    """
    mask = timesteps < cutoff_timestep
    if not torch.any(mask):
        return (x0_pred * 0.0).sum()

    return perceptual_loss(
        x0_pred[mask],
        content_images[mask],
        style_images,
        content_weight=content_weight,
        style_weight=style_weight,
    )
