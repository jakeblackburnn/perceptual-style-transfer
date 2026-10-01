import pytest
import torch

from diffuser.config.unet_presets import get_unet_preset
from diffuser.models import ConditionedUNet


def test_forward_pass_shape():
    unet_config = get_unet_preset("small")
    image_size = 16
    model = ConditionedUNet(unet_config, image_size).eval()

    x_t = torch.rand(1, 3, image_size, image_size)
    content_image = torch.rand(1, 3, image_size, image_size)
    t = torch.tensor([0])

    with torch.no_grad():
        out = model(x_t, content_image, t)

    assert out.shape == (1, 3, image_size, image_size)


def test_indivisible_image_size_raises():
    unet_config = get_unet_preset("small")  # 2 levels -> divisor 2
    with pytest.raises(ValueError):
        ConditionedUNet(unet_config, image_size=15)
