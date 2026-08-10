"""Regression guard for the train/inference preprocessing mismatch.

Training (dataset.py) and inference (style_transfer/inference.py) must
preprocess images identically -- both to a [0, 1]-range tensor via
ToTensor() with no extra scaling. See CLAUDE.md Goals section.
"""

import torch
from PIL import Image

from style_transfer.dataset import ImageDataset
from style_transfer.inference import preprocess as inference_preprocess


def _make_test_image(path, size=(16, 16)):
    img = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            img.putpixel((x, y), ((x * 8) % 256, (y * 8) % 256, 128))
    img.save(path)
    return img


def test_training_and_inference_preprocessing_agree(tmp_path):
    img_path = tmp_path / "sample.png"
    img = _make_test_image(img_path)

    dataset = ImageDataset(str(tmp_path), image_frac=1.0, image_size=16, device="cpu", quiet=True)
    training_tensor = dataset[0]

    inference_tensor = inference_preprocess(img)

    assert training_tensor.shape == inference_tensor.shape
    assert torch.allclose(training_tensor, inference_tensor, atol=1e-6)


def test_preprocessing_output_is_unit_range():
    img = Image.new("RGB", (8, 8), color=(255, 255, 255))
    tensor = inference_preprocess(img)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
