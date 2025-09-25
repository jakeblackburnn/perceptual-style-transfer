import torch.nn as nn

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