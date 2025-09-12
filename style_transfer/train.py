import os
import time
import torch

from torch.utils.data import DataLoader
from style_transfer.dataset import ImageDataset, SingleImageDataset

from style_transfer.loss import vgg_perceptual_loss
from style_transfer.feature_extractors.vgg import initialize_vgg

from style_transfer.image_transformers.small_guy import SmallGuy 
from style_transfer.image_transformers.medium_guy import MediumGuy 
from style_transfer.image_transformers.big_guy import BigGuy 

from utils.metrics import MetricsLogger, save_checkpoint, save_final_model

def train_epoch(model, optimizer, image_loaders, style_weight, device):
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


def train_stage(model, optimizer, image_loaders, stage_idx, stage_config, out_dir, logger, device):

    # set optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = stage_config.get('lr')

    epochs = stage_config.get('epochs', 1)
    style_weight = stage_config.get('style_weight', 1e5)
    for epoch in range(1, epochs + 1):


        avg_loss, elapsed = train_epoch(model, optimizer, image_loaders, style_weight, device)
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



def train_model(model_name, model_config, device):

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


    # Initialize VGG model once for the entire training
    print(f"Initializing VGG model with {layer_preset} preset on {device}")
    initialize_vgg(layer_preset=layer_preset, device=device)


    # TODO: update metrics logger to no longer accept curriculum name
    out_dir = f"models/{model_name}"
    logger = MetricsLogger(logfile=os.path.join(out_dir, 'metrics.csv'), curriculum_name=model_name)

    # create Model object
    if model_size == "small":
        model = SmallGuy();
    elif model_size == "medium":
        model = MediumGuy();
    elif model_size == "big":
        model = BigGuy();
    else: 
        print("""
              something is horribly wrong this is a nightmare 
              please help I couldnt figure out what sized model to use
              """)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e4) # default initial lr

    
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

        image_loaders = (content_loader, style_loader)

        # run training for this stage
        train_stage(
            model=model,
            optimizer=optimizer,
            image_loaders=image_loaders,
            stage_idx=stage_idx,
            stage_config=stage,
            out_dir=out_dir,
            logger=logger,
            device=device
        )

    print("training complete.")
    logger.save_metadata()
    
    final_model_path = os.path.join(out_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
