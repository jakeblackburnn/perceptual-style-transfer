import torch
from pathlib import Path

from style_transfer.img_transformer import Transformer
from style_transfer.train           import train_curriculum

if __name__ == '__main__':

    content_dir = 'images/VOC2012'
    style_dir = 'images/Impressionism'

    content_frac = 0.05 # use random sample of 20% of total images
    style_frac = 1.0    # use all style images

    curriculum = 'dry_run'
    
    # Extract style name from directory path
    style_name = Path(style_dir).name
    
    # Create hierarchical output directory: curriculum/style
    out_dir = f'models/{curriculum}/{style_name}'

    # Device setup
    print("PyTorch version:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    model = Transformer()

    train_curriculum(model, curriculum, out_dir, content_dir, content_frac, style_dir, style_frac, device)
