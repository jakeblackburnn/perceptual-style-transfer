"""stylize_image's size/mode contract and model lookup via model.json."""

import json

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from style_transfer.inference import list_models, load_named_model, stylize_image
from style_transfer.models import StyleTransferModel


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return StyleTransferModel("small").eval()


def _noise_image(size, mode="RGB"):
    torch.manual_seed(1)
    width, height = size
    return ToPILImage()(torch.rand(3, height, width)).convert(mode)


def _write_model(models_dir, name, model_json=None):
    folder = models_dir / name
    folder.mkdir(parents=True)
    torch.save(StyleTransferModel("small").state_dict(), folder / f"{name}.pth")
    if model_json is None:
        model_json = json.dumps({"name": name, "model_size": "small", "weights": f"{name}.pth"})
    (folder / "model.json").write_text(model_json)


@pytest.mark.parametrize("size", [(64, 64), (65, 97), (130, 71), (16, 16)])
def test_output_size_equals_input_size(model, size):
    assert stylize_image(_noise_image(size), model).size == size


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "LA"])
def test_non_rgb_inputs_work(model, mode):
    out = stylize_image(_noise_image((33, 20), mode), model)
    assert out.mode == "RGB"
    assert out.size == (33, 20)


def test_transparent_pixels_are_flattened_onto_white(model):
    transparent = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    white = Image.new("RGB", (32, 32), (255, 255, 255))
    assert stylize_image(transparent, model).tobytes() == stylize_image(white, model).tobytes()


def test_exif_orientation_is_applied(model):
    img = _noise_image((40, 24))
    img.getexif()[0x0112] = 6  # orientation: rotate 90 degrees to display upright
    assert stylize_image(img, model).size == (24, 40)


def test_max_side_downsizes_preserving_aspect(model):
    assert stylize_image(_noise_image((200, 100)), model, max_side=50).size == (50, 25)


def test_max_side_leaves_smaller_images_alone(model):
    assert stylize_image(_noise_image((40, 20)), model, max_side=50).size == (40, 20)


def test_too_small_input_raises(model):
    with pytest.raises(ValueError, match="at least 16"):
        stylize_image(_noise_image((15, 64)), model)


def test_multiple_of_four_input_matches_direct_model_call(model):
    img = _noise_image((64, 48))
    with torch.no_grad():
        direct = model(ToTensor()(img).unsqueeze(0))[0]
    expected = ToPILImage()(direct.clamp(0, 1))
    assert stylize_image(img, model).tobytes() == expected.tobytes()


def test_list_models(tmp_path):
    _write_model(tmp_path, "zebra")
    _write_model(tmp_path, "apple")
    (tmp_path / "no_json").mkdir()
    torch.save({}, tmp_path / "no_json" / "no_json.pth")

    models = list_models(tmp_path)

    assert [m["name"] for m in models] == ["apple", "zebra"]
    assert models[0]["model_size"] == "small"
    assert models[0]["path"] == (tmp_path / "apple" / "apple.pth").resolve()
    assert models[0]["path"].is_absolute()


def test_list_models_missing_dir_is_empty(tmp_path):
    assert list_models(tmp_path / "absent") == []


@pytest.mark.parametrize("broken", ["{not json", json.dumps({"name": "bad", "weights": "bad.pth"})])
def test_broken_model_json_raises_naming_the_file(tmp_path, broken):
    _write_model(tmp_path, "bad", model_json=broken)
    with pytest.raises(ValueError, match="model.json"):
        list_models(tmp_path)


def test_load_named_model(tmp_path):
    _write_model(tmp_path, "apple")
    loaded = load_named_model("apple", tmp_path)
    assert isinstance(loaded, StyleTransferModel)
    assert not loaded.training


def test_load_named_model_unknown_name_lists_available(tmp_path):
    _write_model(tmp_path, "apple")
    with pytest.raises(FileNotFoundError, match="apple"):
        load_named_model("pear", tmp_path)
