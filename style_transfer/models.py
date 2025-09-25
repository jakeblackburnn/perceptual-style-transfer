import torch.nn as nn
from .architectures.residual_block import ResidualBlock

class StyleTransferModel(nn.Module):
    def __init__(self, size_config="medium", in_channels=3, out_channels=3):
        super().__init__()

        # Size configurations
        configs = {
            "small": {
                "ngf": 32,
                "n_residual": 2,
                "initial_kernel_size": 5,
                "initial_padding": 2,
                "final_kernel_size": 5,
                "final_padding": 2
            },
            "medium": {
                "ngf": 32,
                "n_residual": 5,
                "initial_kernel_size": 9,
                "initial_padding": 4,
                "final_kernel_size": 9,
                "final_padding": 4
            },
            "big": {
                "ngf": 64,
                "n_residual": 9,
                "initial_kernel_size": 9,
                "initial_padding": 4,
                "final_kernel_size": 9,
                "final_padding": 4
            }
        }

        if size_config not in configs:
            raise ValueError(f"Unknown size_config: {size_config}. Must be one of: {list(configs.keys())}")

        config = configs[size_config]
        ngf = config["ngf"]
        n_residual = config["n_residual"]

        # Initial conv block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(config["initial_padding"]),
            nn.Conv2d(in_channels, ngf, kernel_size=config["initial_kernel_size"], stride=1),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2, affine=True),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4, affine=True),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual):
            res_blocks.append(ResidualBlock(ngf*4))
        self.residuals = nn.Sequential(*res_blocks)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*4, ngf*2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ngf*2, affine=True),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )

        # Final conv block
        self.final = nn.Sequential(
            nn.ReflectionPad2d(config["final_padding"]),
            nn.Conv2d(ngf, out_channels, kernel_size=config["final_kernel_size"], stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residuals(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        # Scale from [-1,1] to [0,1]
        return (x + 1) / 2


def create_model(size="medium", device="cpu"):
    """Factory function for clean model creation"""
    model = StyleTransferModel(size_config=size).to(device)
    return model
