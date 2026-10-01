"""Named UNet size presets, analogous to style_transfer.models's small/medium/big."""

from .schema import UNetConfig

UNET_PRESETS = {
    "small": UNetConfig(
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    ),
    "standard": UNetConfig(
        layers_per_block=2,
        block_out_channels=(64, 128, 192, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ),
}


def get_unet_preset(name):
    if name not in UNET_PRESETS:
        raise KeyError(f"UNet preset '{name}' not found. Available presets: {list(UNET_PRESETS.keys())}")
    return UNET_PRESETS[name]
