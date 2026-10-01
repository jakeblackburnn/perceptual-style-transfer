import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from style_transfer.dataset import ImageDataset, SingleImageDataset
from style_transfer.feature_extractors.vgg import initialize_vgg

from .models import ConditionedUNet
from .loss import diffusion_perceptual_loss
from .schedule import build_train_scheduler, predict_x0, to_diffusion_space, from_diffusion_space
from .utils import DiffusionMetricsLogger


def train_epoch(model, ema, optimizer, scheduler, image_loaders, cfg, device):
    start_time = time.time()

    model.train()
    mse_losses = []
    perceptual_losses = []
    total_losses = []

    content_loader, style_loader = image_loaders

    # this is hacky for single image mode but it works -- same pattern as style_transfer/train.py
    style_iter = iter(style_loader)

    cutoff_timestep = int(cfg.perceptual_cutoff_frac * cfg.num_train_timesteps)

    for idx, content_batch in enumerate(content_loader, start=1):
        try:
            style_batch = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_batch = next(style_iter)

        content01 = content_batch.to(device)
        style01 = style_batch.to(device)
        content_diff = to_diffusion_space(content01)

        t = torch.randint(0, cfg.num_train_timesteps, (content01.shape[0],), device=device)
        noise = torch.randn_like(content_diff)
        x_t = scheduler.add_noise(content_diff, noise, t)

        eps_pred = model(x_t, content_diff, t)
        mse_loss = F.mse_loss(eps_pred, noise)

        x0_pred01 = from_diffusion_space(predict_x0(scheduler, x_t, t, eps_pred))
        perc_loss = diffusion_perceptual_loss(
            x0_pred01, content01, style01, t, cutoff_timestep,
            content_weight=cfg.perceptual_content_weight, style_weight=cfg.perceptual_style_weight,
        )

        loss = mse_loss + cfg.perceptual_loss_weight * perc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        ema.step(model.parameters())

        mse_losses.append(mse_loss.item())
        perceptual_losses.append(perc_loss.item())
        total_losses.append(loss.item())

        total_batches = len(content_loader)
        if (idx % 10) == 0:
            print(f"batch {idx}/{total_batches} - complete")

    elapsed = time.time() - start_time

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0

    return avg(mse_losses), avg(perceptual_losses), avg(total_losses), elapsed


def train_model(experiment_name, cfg, device):
    print(f"training {experiment_name}:")
    print(cfg)

    print(f"Initializing VGG model with {cfg.layer_preset} preset on {device}")
    initialize_vgg(layer_preset=cfg.layer_preset, device=device)

    out_dir = f"models/{experiment_name}"
    logger = DiffusionMetricsLogger(logfile=os.path.join(out_dir, 'metrics.csv'), curriculum_name=experiment_name)

    model = ConditionedUNet(cfg.unet, cfg.image_size).to(device)
    ema = EMAModel(model.parameters(), decay=cfg.ema_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = build_train_scheduler(cfg)

    cdir = cfg.content.get('dataset')
    cfrac = cfg.content.get('fraction', 1)
    single_style = cfg.style.get('single', False)
    sdir = cfg.style.get('dataset')
    sfrac = cfg.style.get('fraction', 1)

    content_dataset = ImageDataset(cdir, cfrac, cfg.image_size, device, quiet=False)
    if single_style:
        style_dataset = SingleImageDataset(sdir, cfg.image_size, device, quiet=False)
    else:
        style_dataset = ImageDataset(sdir, sfrac, cfg.image_size, device, quiet=False)

    dataset_info = {'content': content_dataset.dataset_info, 'style': style_dataset.dataset_info}
    logger.save_stage_config({'epochs': cfg.epochs, 'lr': cfg.lr}, dataset_info)

    style_batch_size = 1 if single_style else cfg.content_batch_size

    content_loader = DataLoader(
        content_dataset, batch_size=cfg.content_batch_size, shuffle=True,
        num_workers=2, persistent_workers=True,
    )
    style_loader = DataLoader(
        style_dataset, batch_size=style_batch_size, shuffle=True,
        num_workers=2, persistent_workers=True,
    )

    stage_idx = 1  # no multi-stage curriculum here, but the "stage" column keeps the metrics.csv schema consistent
    for epoch in range(1, cfg.epochs + 1):
        mse_loss, perc_loss, total_loss, elapsed = train_epoch(
            model, ema, optimizer, scheduler, (content_loader, style_loader), cfg, device,
        )
        print(f"epoch {epoch}: mse={mse_loss:.4f} perceptual={perc_loss:.4f} total={total_loss:.4f}, time={elapsed:.1f}s")
        logger.log({
            'stage': stage_idx, 'epoch': epoch,
            'mse_loss': mse_loss, 'perceptual_loss': perc_loss, 'loss': total_loss, 'time': elapsed,
        })

    print("training complete, saving metrics ...")
    logger.save()
    logger.save_metadata()

    # copy EMA weights into the model before the final save, so inference.py
    # needs no EMA-aware branching to load the final checkpoint
    ema.copy_to(model.parameters())
    final_model_path = os.path.join(out_dir, f"{experiment_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    for loader in (content_loader, style_loader):
        if hasattr(loader, '_iterator') and loader._iterator is not None:
            del loader._iterator

    if device.type in ['cuda', 'mps']:
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    import style_transfer.feature_extractors.vgg as vgg_module
    vgg_module._vgg_model = None
