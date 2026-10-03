"""Single source of truth for running a trained StyleTransferModel on an image.

Contract of stylize_image:

- Preprocessing stays in lockstep with the training pipeline in dataset.py:
  plain ToTensor() (float, [0, 1] range). Do not scale by 255 before the
  forward pass -- models.py's final activation is a tanh rescaled to [0, 1],
  trained end-to-end against [0, 1]-range inputs.
- EXIF orientation is applied and the image is converted to RGB
  (transparency is flattened onto white).
- With max_side, an image whose longer side exceeds it is downsized first,
  preserving aspect ratio.
- The output has exactly the size of the (possibly downsized) input. The
  generator downsamples twice, so the tensor is reflect-padded on the right
  and bottom to a multiple of four and the result is cropped back. An image
  whose sides are already multiples of four reaches the model unchanged.
- Images smaller than MIN_SIDE pixels on a side are rejected.

Models are looked up by name: <models_dir>/<name>/model.json describes the
model and names its weights file. The architecture size always comes from
model.json, never from the training config.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from style_transfer.models import StyleTransferModel

DEFAULT_MODELS_DIR = Path("artifacts/models")
MIN_SIDE = 16
REQUIRED_MODEL_KEYS = ("name", "model_size", "weights")

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

postprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.clamp(0, 1)),
    transforms.ToPILImage(),
])


def list_models(models_dir=DEFAULT_MODELS_DIR):
    """Return one dict per <models_dir>/*/model.json, sorted by name.

    Each dict is the parsed model.json plus "path", the absolute path of the
    weights file. Directories without a model.json are not models.
    """
    models = []
    for model_json in Path(models_dir).glob("*/model.json"):
        try:
            info = json.loads(model_json.read_text())
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot read {model_json}: {e}") from e
        if not isinstance(info, dict):
            raise ValueError(f"{model_json} must contain a JSON object")
        missing = [key for key in REQUIRED_MODEL_KEYS if key not in info]
        if missing:
            raise ValueError(f"{model_json} lacks required key(s): {', '.join(missing)}")
        info["path"] = (model_json.parent / info["weights"]).resolve()
        models.append(info)
    return sorted(models, key=lambda info: info["name"])


def load_model(model_path, model_size, device="cpu"):
    """Load a trained StyleTransferModel from a checkpoint file."""
    model = StyleTransferModel(size_config=model_size).to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


def load_named_model(name, models_dir=DEFAULT_MODELS_DIR, device="cpu"):
    """Load the model called `name` from models_dir, sized per its model.json."""
    models = list_models(models_dir)
    for info in models:
        if info["name"] == name:
            return load_model(info["path"], info["model_size"], device)
    available = ", ".join(info["name"] for info in models) or "none"
    raise FileNotFoundError(f"no model named '{name}' in {models_dir} (available: {available})")


def _to_rgb(img):
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return img.convert("RGB")


def stylize_image(content_img: Image.Image, model: StyleTransferModel, device="cpu", max_side=None) -> Image.Image:
    """Run style transfer on a PIL image; the result has the input's size
    (or the downsized size when max_side applies). See the module docstring."""
    img = _to_rgb(content_img)

    width, height = img.size
    if max_side is not None and max(width, height) > max_side:
        scale = max_side / max(width, height)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
        img = img.resize((width, height), Image.LANCZOS)

    if min(width, height) < MIN_SIDE:
        raise ValueError(f"image is {width}x{height}; both sides must be at least {MIN_SIDE} px")

    content_tensor = preprocess(img).unsqueeze(0).to(device)
    # MIN_SIDE is far above the 3 px maximum pad, so reflect padding is always valid.
    pad_right, pad_bottom = -width % 4, -height % 4
    if pad_right or pad_bottom:
        content_tensor = F.pad(content_tensor, (0, pad_right, 0, pad_bottom), mode="reflect")

    with torch.no_grad():
        output_tensor = model(content_tensor)
    return postprocess(output_tensor[0, :, :height, :width].cpu())


def select_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoint(experiment_name):
    """Locate <base>/<name>/<name>.pth under artifacts/models/ or models/.

    Unused by style transfer itself (models are found via model.json). It
    stays only because diffuser/inference.py imports it from here; delete it
    once that import is gone.
    """
    for base in (Path("artifacts/models"), Path("models")):
        candidate = base / experiment_name / f"{experiment_name}.pth"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found for experiment '{experiment_name}' under "
        f"artifacts/models/ or models/"
    )
