import time
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import style_transfer.config as config

from style_transfer.loss    import vgg_perceptual_loss, set_vgg_device
from style_transfer.dataset import StyleTransferDataset

from utils.metrics import MetricsLogger, save_checkpoint, save_final_model

def train_epoch(model, optimizer, data_loader, style_weight, device):

    start_time = time.time() # start timer

    model.train()
    losses = []
    for idx, (content, style) in enumerate(data_loader, start=1):

        content = content.to(device)
        style = style.to(device)

        optimizer.zero_grad()
        loss = vgg_perceptual_loss(model, (content, style), style_weight=style_weight)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        total_batches = len(data_loader)
        if (idx % 10) == 0:
            print(f"batch {idx}/{total_batches} - complete")

    elapsed = time.time() - start_time # end timer

    avg_loss = sum(losses) / len(losses) if losses else 0
    return avg_loss, elapsed



def train_stage(model, data_loader, stage_idx, stage_config, out_dir, logger, device):

    # build Adam Optimizer w/ specified LR
    optimizer = config.optimizer(model.parameters(), stage_config.get('lr'))

    epochs = stage_config.get('epochs', 1)
    style_weight = stage_config.get('style_weight', 1e5)
    for epoch in range(1, epochs + 1):


        avg_loss, elapsed = train_epoch(model, optimizer, data_loader, style_weight, device)
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


def collate_fn(batch):
    contents = [item[0] for item in batch]
    styles = [item[1] for item in batch]
    return default_collate(contents), default_collate(styles)

def train_curriculum(model, curriculum_name, out_dir, content_dir, cfrac, style_dir, sfrac, device):

    curriculum = config.Curricula.get(curriculum_name)
    if curriculum is None:
        raise ValueError(f"Unknown curriculum '{curriculum_name}'.")

    logger = MetricsLogger(logfile=os.path.join(out_dir, "metrics.csv"), curriculum_name=curriculum_name)

    set_vgg_device(device)
    model.to(device)

    
    for stage_idx, stage in enumerate(curriculum['stages'], start=1):

        resolution = stage.get('resolution')
        batch_size = stage.get('batch_size')
        print(f"\n=== Starting stage {stage_idx} at resolution {resolution} ===")

        # Create dataset for this stage (may have different resolution)
        dataset = StyleTransferDataset(content_dir, cfrac, style_dir, sfrac, resolution, device)
        
        # Save stage config and dataset info (dataset info saved only once)
        logger.save_stage_config(stage, dataset.dataset_info)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=collate_fn
        )

        # run training for this stage
        train_stage(
            model=model,
            data_loader=data_loader,
            stage_idx=stage_idx,
            stage_config=stage,
            out_dir=out_dir,
            logger=logger,
            device=device
        )

    print("training complete.")
    logger.save_metadata()

