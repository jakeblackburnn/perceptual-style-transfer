import torch

from style_transfer.config import Models
from style_transfer.train import train_model

if __name__ == '__main__':

    model_name = 'kanagawa_dry_run'
    model_config = Models.get(model_name)

    print("PyTorch version:", torch.__version__)

    # Device setup
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("using device:", device)

    train_model(model_name, model_config, device)
