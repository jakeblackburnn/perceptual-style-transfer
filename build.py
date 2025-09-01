import torch

from style_transfer.config import Models
from style_transfer.train import train_model

if __name__ == '__main__':

    model_name = 'medium_no_style_deep_loss_v1'
    model_config = Models.get(model_name)

    # Device setup
    print("PyTorch version:", torch.__version__)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("using device:", device)

    train_model(model_name, model_config, device)
