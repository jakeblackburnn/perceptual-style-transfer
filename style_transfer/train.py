import time
import os

import torch
from torch.utils.data import DataLoader

import style_transfer.config as config

from style_transfer.loss    import vgg_perceptual_loss, set_vgg_device
from style_transfer.dataset import ImageDataset

from utils.metrics import MetricsLogger, save_checkpoint, save_final_model

def train_epoch(model, optimizer, content_loader, style_loader, style_weight, device):

    start_time = time.time() # start timer

    model.train()
    losses = []
    
    # Create iterator for style batches that can be reset
    style_iter = iter(style_loader)
    
    for idx, content_batch in enumerate(content_loader, start=1):
        # Get next style batch, reset iterator if exhausted
        try:
            style_batch = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_batch = next(style_iter)

        content_batch = content_batch.to(device)
        style_batch = style_batch.to(device)

        optimizer.zero_grad()
        loss = vgg_perceptual_loss(model, content_batch, style_batch, style_weight=style_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        total_batches = len(content_loader)
        if (idx % 10) == 0:
            print(f"batch {idx}/{total_batches} - complete")

    elapsed = time.time() - start_time # end timer

    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss, elapsed



def train_stage(model, content_loader, style_loader, stage_idx, stage_config, out_dir, logger, device):

    # build Adam Optimizer w/ specified LR
    optimizer = config.optimizer(model.parameters(), stage_config.get('lr'))

    epochs = stage_config.get('epochs', 1)
    style_weight = stage_config.get('style_weight', 1e5)
    for epoch in range(1, epochs + 1):


        avg_loss, elapsed = train_epoch(model, optimizer, content_loader, style_loader, style_weight, device)
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



def train_curriculum(model, curriculum_name, out_dir, content_dir, cfrac, style_dir, sfrac, device):

    curriculum = config.Curricula.get(curriculum_name)
    if curriculum is None:
        raise ValueError(f"Unknown curriculum '{curriculum_name}'.")

    logger = MetricsLogger(logfile=os.path.join(out_dir, "metrics.csv"), curriculum_name=curriculum_name)

    set_vgg_device(device)
    model.to(device)

    
    for stage_idx, stage in enumerate(curriculum['stages'], start=1):

        resolution = stage.get('resolution')
        print(f"\n=== Starting stage {stage_idx} at resolution {resolution} ===")

        # Create separate datasets for content and style
        content_dataset = ImageDataset(content_dir, cfrac, resolution, device, quiet=False)
        style_dataset = ImageDataset(style_dir, sfrac, resolution, device, quiet=False)
        
        # Save stage config and dataset info
        combined_dataset_info = {
            'content': content_dataset.dataset_info,
            'style': style_dataset.dataset_info
        }
        logger.save_stage_config(stage, combined_dataset_info)
        
        # Get batch shape configuration with defaults
        content_batch_size = stage.get('content_batch_size', 1)
        style_batch_size = stage.get('style_batch_size', 4)
        
        # Create separate data loaders
        content_loader = DataLoader(
            content_dataset,
            batch_size=content_batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )
        
        style_loader = DataLoader(
            style_dataset,
            batch_size=style_batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )

        # run training for this stage
        train_stage(
            model=model,
            content_loader=content_loader,
            style_loader=style_loader,
            stage_idx=stage_idx,
            stage_config=stage,
            out_dir=out_dir,
            logger=logger,
            device=device
        )

    print("training complete.")
    logger.save_metadata()

