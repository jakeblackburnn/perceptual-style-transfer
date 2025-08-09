import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class StyleTransferDataset(Dataset):

    def __init__(self, content_dir, content_frac, style_dir, style_frac, image_size, device, quiet=False):

        valid_exts = {".jpg", ".jpeg", ".png"}

        # recursively grab all image files
        self.content_paths = [
            str(p) for p in Path(content_dir).rglob("*")
            if p.suffix.lower() in valid_exts
        ]
        self.style_paths = [
            str(p) for p in Path(style_dir).rglob("*")
            if p.suffix.lower() in valid_exts
        ]

        # store original counts for reporting
        total_content = len(self.content_paths)
        total_style = len(self.style_paths)

            # ----- sanity checks -----

        # check images exist
        if not self.content_paths:
            raise ValueError(f"No content images found in {content_dir}")
        if not self.style_paths:
            raise ValueError(f"No style images found in {style_dir}")

        # check fraction of content images makes sense
        if (0 < content_frac) and (content_frac <= 1.0):
            k = int(len(self.content_paths) * content_frac)
            self.content_paths = random.sample(self.content_paths, k)
        else:
            print("Dataset init: content fraction is nonsensical")

        # check fraction of style images makes sense
        if (0 < style_frac) and (style_frac <= 1.0):
            k = int(len(self.style_paths) * style_frac)
            self.style_paths = random.sample(self.style_paths, k)
        else:
            print("Dataset init: style fraction is nonsensical")

        # store final counts
        used_content = len(self.content_paths)
        used_style = len(self.style_paths)

        # display dataset information
        style_name = Path(style_dir).name
        content_name = Path(content_dir).name
        
        if not quiet:
            print(f"\n=== Dataset Information ===")
            print(f"Content dataset: {content_name}")
            print(f"  Total images found: {total_content}")
            print(f"  Fraction used: {content_frac}")
            print(f"  Images used: {used_content}")
            print(f"Style dataset: {style_name}")
            print(f"  Total images found: {total_style}")
            print(f"  Fraction used: {style_frac}")
            print(f"  Images used: {used_style}")
            print(f"============================\n")

        # store dataset info for access by training pipeline
        self.dataset_info = {
            'content_name': content_name,
            'style_name': style_name,
            'total_content': total_content,
            'total_style': total_style,
            'content_frac': content_frac,
            'style_frac': style_frac,
            'used_content': used_content,
            'used_style': used_style
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
        return len(self.content_paths)


    def __getitem__(self, idx: int):
        # Load and transform content image
        content_path = self.content_paths[idx]
        content_img = Image.open(content_path).convert("RGB")
        content_tensor = self.transform(content_img)

        # Randomly sample a style image
        style_idx = random.randrange(len(self.style_paths))
        style_path = self.style_paths[style_idx]
        style_img = Image.open(style_path).convert("RGB")
        style_tensor = self.transform(style_img)

        return content_tensor, style_tensor
