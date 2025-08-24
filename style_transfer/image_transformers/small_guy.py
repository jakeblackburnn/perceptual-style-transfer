import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# small image transformer
# 2 residual blocks, 5x5 kernels for ******

class SmallGuy(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, ngf=32, n_residual=2):
        super().__init__()

        # Initial conv block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, ngf, kernel_size=5, stride=1),
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
            nn.ReflectionPad2d(2),
            nn.Conv2d(ngf, out_channels, kernel_size=5, stride=1),
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
