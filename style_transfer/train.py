import json
import os
import subprocess
import time
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from style_transfer.dataset import ImageDataset, SingleImageDataset

from style_transfer.models import StyleTransferModel 
from style_transfer.feature_extractors.vgg import initialize_vgg
from style_transfer.config.layer_presets import get_layer_preset
from style_transfer.loss import perceptual_loss
from style_transfer.utils.metrics import MetricsLogger, save_checkpoint

# defaults for values a curriculum stage may leave out
DEFAULT_STYLE_WEIGHT = 2.5  # sized for the sum-reduced style loss, see loss.py
DEFAULT_TV_WEIGHT = 0.0

def train_epoch(model, optimizer, image_loaders, style_weight, tv_weight, device):
    start_time = time.time()

    model.train()
    losses = []

    content_loader, style_loader = image_loaders

    # this is hacky for single image mode but it works
    style_iter = iter(style_loader) # resettable style image iterator

    for idx, content_batch in enumerate(content_loader, start=1):
        try:
            style_batch = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_batch = next(style_iter)

        content_batch = content_batch.to(device)
        style_batch = style_batch.to(device)

        optimizer.zero_grad()
        generated_images = model(content_batch)
        loss = perceptual_loss(generated_images, content_batch, style_batch,
                               style_weight=style_weight, tv_weight=tv_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        total_batches = len(content_loader)
        if (idx % 10) == 0:
            print(f"batch {idx}/{total_batches} - complete")

    elapsed = time.time() - start_time # end timer

    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss, elapsed, len(losses) # len(losses) = optimizer steps taken


def train_stage(model, optimizer, image_loaders, stage_idx, stage_config, out_dir, logger, device):

    # set optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = stage_config.get('lr')

    epochs = stage_config.get('epochs', 1)
    style_weight = stage_config.get('style_weight', DEFAULT_STYLE_WEIGHT)
    tv_weight = stage_config.get('tv_weight', DEFAULT_TV_WEIGHT)
    steps = 0
    for epoch in range(1, epochs + 1):

        avg_loss, elapsed, epoch_steps = train_epoch(model, optimizer, image_loaders, style_weight, tv_weight, device)
        steps += epoch_steps
        print(f"epoch {epoch}: avg loss = {avg_loss}, time = {elapsed} s")
        log_data = {
            'stage': stage_idx,
            'epoch': epoch,
            'loss': avg_loss,
            'time': elapsed,
        }
        logger.log(log_data)

        ckpt_dir = os.path.join(out_dir, "checkpoints");
        save_checkpoint(model, optimizer, epoch, stage_idx, ckpt_dir)

    print(f"stage {stage_idx} complete, saving metrics ...")
    logger.save()

    return steps


def git_commit():
    # short hash of the code that trained the model, or None outside a git checkout
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_model(model_name, model_config, device, models_dir="artifacts/models"):
    """Train one experiment and return the Path of its final weights.

    Everything is written to <models_dir>/<model_name>/: metrics.csv,
    checkpoints/, <model_name>.pth (a bare state_dict) and model.json, which
    records what the weights were trained with.
    """

    print(f"training {model_name}:")
    print(model_config)

    # unpack training configuration
    curriculum = model_config.get('curriculum')
    model_size = model_config.get('model_size', "medium")
    layer_preset = model_config.get('layer_preset', 'standard')

    cdir  = model_config.get('content', {}).get('dataset')
    cfrac = model_config.get('content', {}).get('fraction', 1) # default to entire dir

    single_style = model_config.get('style', {}).get('single', False)

    sdir  = model_config.get('style', {}).get('dataset')
    sfrac = model_config.get('style', {}).get('fraction', 1) # default to entire dir

    num_workers = model_config.get('num_workers', 2) # DataLoader workers; 0 loads in-process

    # Initialize VGG model once for the entire training
    print(f"Initializing VGG model with {layer_preset} preset on {device}")
    initialize_vgg(layer_preset=layer_preset, device=device)


    # TODO: update metrics logger to no longer accept curriculum name
    out_dir = Path(models_dir) / model_name
    logger = MetricsLogger(logfile=os.path.join(out_dir, 'metrics.csv'), curriculum_name=model_name)

    # create Model object
    model = StyleTransferModel(size_config=model_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4) # default initial lr

    all_loaders = []  # Track all DataLoaders for cleanup
    resolved_stages = []  # stages with defaults filled in, for model.json
    steps = 0

    for stage_idx, stage in enumerate(curriculum['stages'], start=1):

        resolution = stage.get('res', 256)
        print(f"\n=== Starting stage {stage_idx} at resolution {resolution} ===")

        content_dataset = ImageDataset(cdir, cfrac, resolution, device, quiet=False)

        if single_style == True:
            style_dataset = SingleImageDataset(sdir, resolution, device, quiet=False)
        else:
            style_dataset = ImageDataset(sdir, sfrac, resolution, device, quiet=False)
        
        dataset_info = {
            'content': content_dataset.dataset_info,
            'style': style_dataset.dataset_info
        }
        logger.save_stage_config(stage, dataset_info)
        
        # Get batch shape configuration with defaults
        content_batch_size = stage.get('content_batch_size', 1)
        style_batch_size = stage.get('style_batch_size', 4)

        
        content_loader = DataLoader(
            content_dataset,
            batch_size=content_batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )
        
        style_loader = DataLoader(
            style_dataset,
            batch_size=style_batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )

        image_loaders = (content_loader, style_loader)
        all_loaders.extend([content_loader, style_loader])  # Track loaders

        # run training for this stage
        steps += train_stage(
            model=model,
            optimizer=optimizer,
            image_loaders=image_loaders,
            stage_idx=stage_idx,
            stage_config=stage,
            out_dir=out_dir,
            logger=logger,
            device=device
        )

        resolved_stages.append({
            'res': resolution,
            'epochs': stage.get('epochs', 1),
            'lr': stage.get('lr'),
            'style_weight': stage.get('style_weight', DEFAULT_STYLE_WEIGHT),
            'tv_weight': stage.get('tv_weight', DEFAULT_TV_WEIGHT),
            'content_batch_size': content_batch_size,
            'style_batch_size': style_batch_size,
        })

    print("training complete.")
    logger.save_metadata()

    final_model_path = out_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # model.json: what these weights are and how they were trained
    preset = get_layer_preset(layer_preset)
    model_info = {
        'name': model_name,
        'model_size': model_size,
        'weights': final_model_path.name,
        'loss': 'johnson-sum',
        'layer_preset': {
            'name': layer_preset,
            'style_layers': preset['style_layers'],
            'content_layer': preset['content_layer'],
            'style_layer_weights': preset['style_layer_weights'],
            'use_raw_features': preset['use_raw_features'],
        },
        'content_weight': 1.0,
        'stages': resolved_stages,
        'style': model_config.get('style', {}),
        'content': {
            'dataset': cdir,
            'fraction': cfrac,
            'used_images': content_dataset.dataset_info['used_images'],
        },
        'steps': steps,
        'commit': git_commit(),
    }
    with open(out_dir / 'model.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print("Cleaning up DataLoader workers...")
    # Proper cleanup: let PyTorch handle worker shutdown automatically
    for loader in all_loaders:
        # Only delete iterator if it exists, PyTorch handles the rest
        if hasattr(loader, '_iterator') and loader._iterator is not None:
            del loader._iterator
    # Clear the loader references to trigger proper cleanup
    all_loaders.clear()

    # Clear GPU memory and global models
    print("Clearing GPU memory...")
    if device.type in ['cuda', 'mps']:
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print("GPU memory cleared.")

    # Clear global VGG model
    print("Clearing global VGG model...")
    from style_transfer.feature_extractors.vgg import _vgg_model
    import style_transfer.feature_extractors.vgg as vgg_module
    vgg_module._vgg_model = None
    print("VGG model cleared.")

    return final_model_path
