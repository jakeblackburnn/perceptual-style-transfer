import torch

from style_transfer.img_transformer import Transformer
from style_transfer.train           import train_curriculum

if __name__ == '__main__':

    content_dir = 'images/VOC2012/JPEGImages'
    style_dir = 'images/Ukiyo_e'

    content_frac = 0.02 # use random sample of 2% of total images
    style_frac = 0.2    # use random sample of 20% of style images

    curriculum = 'basic_small_batch'
    out_dir = f'models/{curriculum}'

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
