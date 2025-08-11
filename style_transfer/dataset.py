import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):

    def __init__(self, image_dir, image_frac, image_size, device, quiet=False):

        valid_exts = {".jpg", ".jpeg", ".png"}

        # recursively grab all image files
        self.image_paths = [
            str(p) for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in valid_exts
        ]

        # store original count for reporting
        total_images = len(self.image_paths)

        # sanity check
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        # check fraction of images makes sense
        if (0 < image_frac) and (image_frac <= 1.0):
            k = int(len(self.image_paths) * image_frac)
            self.image_paths = random.sample(self.image_paths, k)
        else:
            print("Dataset init: image fraction is nonsensical")

        # store final count
        used_images = len(self.image_paths)

        # display dataset information
        dataset_name = Path(image_dir).name
        
        if not quiet:
            print(f"\n=== Dataset Information ===")
            print(f"Dataset: {dataset_name}")
            print(f"  Total images found: {total_images}")
            print(f"  Fraction used: {image_frac}")
            print(f"  Images used: {used_images}")
            print(f"============================\n")

        # store dataset info for access by training pipeline
        self.dataset_info = {
            'dataset_name': dataset_name,
            'total_images': total_images,
            'image_frac': image_frac,
            'used_images': used_images
        }

        # flags
        self.image_size = image_size
        self.device = device

        # transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    # end init



    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx: int):
        # Load and transform image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor
