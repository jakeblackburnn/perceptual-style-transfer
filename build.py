import torch

from style_transfer.config import Models
from style_transfer.train import train_model

if __name__ == '__main__':

    model_name = 'dry_run'
    model_config = Models.get(model_name)

    # Device setup
    print("PyTorch version:", torch.__version__)
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    train_model(model_name, model_config, device)
