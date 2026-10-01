import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionedUNet(nn.Module):
    """UNet2DModel wrapper implementing channel-concat content conditioning.

    The noisy sample x_t (3ch) and the content image (3ch) are concatenated
    on the channel axis before the forward pass, so the network always has
    direct access to what it's denoising towards. UNet2DModel supports this
    natively via independent in_channels/out_channels -- no monkey-patching
    needed. Unlike the exploratory notebook's hand-rolled TinyUNet, this
    UNet keeps every skip connection between its down/up block pairs and can
    include attention blocks at low resolution.
    """

    def __init__(self, unet_config, image_size: int):
        super().__init__()

        n_levels = len(unet_config.block_out_channels)
        divisor = 2 ** (n_levels - 1)
        if image_size % divisor != 0:
            raise ValueError(
                f"image_size={image_size} is not divisible by {divisor} "
                f"(2^(len(block_out_channels)-1)={n_levels - 1}); UNet2DModel requires this."
            )

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=6,
            out_channels=3,
            layers_per_block=unet_config.layers_per_block,
            block_out_channels=unet_config.block_out_channels,
            down_block_types=unet_config.down_block_types,
            up_block_types=unet_config.up_block_types,
        )

    def forward(self, x_t: torch.Tensor, content_image: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        model_input = torch.cat([x_t, content_image], dim=1)
        return self.unet(model_input, t).sample
