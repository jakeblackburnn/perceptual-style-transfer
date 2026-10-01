"""Fast end-to-end training smoke test.

Targets the exploratory notebook's actual observed failure mode: the
noise-prediction MSE loss trending up instead of down, and the perceptual
loss oscillating without ever producing a finite, sane value. This doesn't
assert full convergence (too flaky for a fast test) -- just that every loss
component stays finite across a few real training steps, on a real (if
tiny) multi-image DataLoader-driven setup rather than the notebook's
single-image overfit.

Uses train_epoch directly rather than train_model, since train_model writes
checkpoints/metrics to a hardcoded models/<name>/ path rather than tmp_path.
"""

import math

import torch
from PIL import Image
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from style_transfer.dataset import ImageDataset, SingleImageDataset
from style_transfer.feature_extractors.vgg import initialize_vgg

from diffuser.config.schema import DiffusionExperimentConfig
from diffuser.config.unet_presets import get_unet_preset
from diffuser.models import ConditionedUNet
from diffuser.schedule import build_train_scheduler
from diffuser.train import train_epoch

initialize_vgg(layer_preset="standard", device="cpu")


def _make_content_dir(tmp_path, n=2, size=(32, 32)):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    for i in range(n):
        img = Image.new("RGB", size, color=(20 * i, 100, 200 - 20 * i))
        img.save(content_dir / f"img{i}.png")
    return content_dir


def _make_style_image(tmp_path, size=(32, 32)):
    style_path = tmp_path / "style.png"
    Image.new("RGB", size, color=(200, 50, 50)).save(style_path)
    return style_path


def test_train_epoch_losses_stay_finite(tmp_path):
    cfg = DiffusionExperimentConfig(
        image_size=32,  # VGG19 needs at least ~32px input for its 5 max-pool stages
        unet=get_unet_preset("small"),
        num_train_timesteps=50,
        content_batch_size=2,
    )

    content_dir = _make_content_dir(tmp_path)
    style_path = _make_style_image(tmp_path)

    content_dataset = ImageDataset(str(content_dir), 1.0, cfg.image_size, "cpu", quiet=True)
    style_dataset = SingleImageDataset(str(style_path), cfg.image_size, "cpu", quiet=True)

    content_loader = DataLoader(content_dataset, batch_size=cfg.content_batch_size, shuffle=True)
    style_loader = DataLoader(style_dataset, batch_size=1, shuffle=True)

    device = torch.device("cpu")
    model = ConditionedUNet(cfg.unet, cfg.image_size).to(device)
    ema = EMAModel(model.parameters(), decay=cfg.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = build_train_scheduler(cfg)

    for _ in range(3):
        mse_loss, perc_loss, total_loss, _elapsed = train_epoch(
            model, ema, optimizer, scheduler, (content_loader, style_loader), cfg, device,
        )
        assert math.isfinite(mse_loss)
        assert math.isfinite(perc_loss)
        assert math.isfinite(total_loss)
