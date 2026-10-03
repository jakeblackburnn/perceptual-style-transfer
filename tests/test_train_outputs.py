"""train_model end to end on a tiny throwaway dataset.

Checks where the outputs land and what model.json records, not training
quality. VGG is replaced by a small stub so no pretrained weights are needed.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import style_transfer.feature_extractors.vgg as vgg_module
import style_transfer.train as train_module
from style_transfer.train import train_model


class FakeFeatureExtractor(nn.Module):
    preset_config = {'style_layers': ['s1', 's2'], 'content_layer': 'c', 'use_raw_features': False}

    def forward(self, x):
        return {'s1': x, 's2': F.avg_pool2d(x, 2), 'c': x}


def _fake_initialize_vgg(layer_preset='standard', device='cpu'):
    vgg_module._vgg_model = FakeFeatureExtractor()


def _save_random_image(path, size=32):
    pixels = (torch.rand(size, size, 3) * 255).to(torch.uint8).numpy()
    Image.fromarray(pixels).save(path)


def test_train_model_writes_weights_and_model_json(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "initialize_vgg", _fake_initialize_vgg)
    # train_model clears the global VGG when it finishes; put back whatever was there
    monkeypatch.setattr(vgg_module, "_vgg_model", vgg_module._vgg_model)
    # any stray relative path (like the old hardcoded models/) would land in tmp_path
    monkeypatch.chdir(tmp_path)

    content_dir = tmp_path / "content"
    content_dir.mkdir()
    for i in range(8):
        _save_random_image(content_dir / f"img{i}.png")
    style_path = tmp_path / "style.png"
    _save_random_image(style_path)

    models_dir = tmp_path / "trained"

    stage = {"res": 32, "epochs": 1, "lr": 5e-4, "style_weight": 2.5, "tv_weight": 1e-6,
             "content_batch_size": 4, "style_batch_size": 1}
    config = {
        "model_size": "small",
        "layer_preset": "standard",
        "content": {"dataset": str(content_dir), "fraction": 1.0},
        "style": {"dataset": str(style_path), "single": True},
        "curriculum": {"stages": [stage]},
        "num_workers": 0,
    }

    weights_path = train_model("tiny", config, torch.device("cpu"), models_dir=models_dir)

    assert weights_path == models_dir / "tiny" / "tiny.pth"
    assert weights_path.exists()
    assert (models_dir / "tiny" / "metrics.csv").exists()
    assert not (tmp_path / "models").exists()

    # the final weights stay a bare state_dict
    state_dict = torch.load(weights_path)
    assert all(isinstance(v, torch.Tensor) for v in state_dict.values())

    model_info = json.loads((models_dir / "tiny" / "model.json").read_text())
    assert set(model_info) == {
        "name", "model_size", "weights", "loss", "layer_preset", "content_weight",
        "stages", "style", "content", "steps", "commit",
    }
    assert model_info["name"] == "tiny"
    assert model_info["model_size"] == "small"
    assert model_info["weights"] == "tiny.pth"
    assert model_info["loss"] == "johnson-sum"
    assert model_info["layer_preset"] == {
        "name": "standard",
        "style_layers": ["0", "5", "10", "19", "28"],
        "content_layer": "21",
        "style_layer_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
        "use_raw_features": False,
    }
    assert model_info["content_weight"] == 1.0
    assert model_info["stages"] == [stage]
    assert model_info["style"] == {"dataset": str(style_path), "single": True}
    assert model_info["content"] == {"dataset": str(content_dir), "fraction": 1.0, "used_images": 8}
    assert model_info["steps"] == 2  # 8 images / batch 4, one epoch
