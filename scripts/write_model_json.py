"""One-off: write model.json next to weights trained before model.json existed.

    python scripts/write_model_json.py [--models-dir artifacts/models] NAME [NAME ...]

Each NAME needs <models_dir>/<NAME>/<NAME>.pth and an experiment of the same
name in style_transfer.config.Models. The model size is checked by loading
the weights, so the model.json it writes is one inference can trust.
"""

import argparse
import json
import sys
from pathlib import Path

# Runnable from a checkout without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from style_transfer.config import Models
from style_transfer.config.layer_presets import get_layer_preset
from style_transfer.models import StyleTransferModel

SIZES = ["small", "medium", "big"]


def fitting_size(weights_path, expected_size):
    """Return the model size the weights load into (strictly), or None."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    for size in [expected_size] + [s for s in SIZES if s != expected_size]:
        try:
            StyleTransferModel(size_config=size).load_state_dict(state_dict, strict=True)
        except RuntimeError:
            continue
        return size
    return None


def write_model_json(name, models_dir, force):
    """Write <models_dir>/<name>/model.json. Returns an error message, or None."""
    weights_path = models_dir / name / f"{name}.pth"
    model_json = models_dir / name / "model.json"

    if not weights_path.is_file():
        return f"{weights_path} does not exist"
    if name not in Models:
        return f"no experiment named '{name}' in style_transfer.config.Models"
    if model_json.exists() and not force:
        return f"{model_json} already exists (pass --force to replace it)"

    config = Models[name]
    config_size = config.get("model_size", "medium")
    size = fitting_size(weights_path, config_size)
    if size is None:
        return f"{weights_path} does not load into any model size ({', '.join(SIZES)})"
    if size != config_size:
        print(f"{name}: config says '{config_size}' but the weights are '{size}'; writing '{size}'")

    preset_name = config.get("layer_preset", "standard")
    try:
        preset = get_layer_preset(preset_name)
    except KeyError as e:
        return f"cannot resolve layer preset: {e}"
    style_layers = preset["style_layers"]

    info = {
        "name": name,
        "model_size": size,
        "weights": weights_path.name,
        "loss": "legacy-mean",
        "layer_preset": {
            "name": preset_name,
            "style_layers": style_layers,
            "content_layer": preset["content_layer"],
            "style_layer_weights": preset.get("style_layer_weights") or [1.0] * len(style_layers),
            "use_raw_features": preset.get("use_raw_features", False),
        },
        "content_weight": 1.0,  # the archived models were all trained with perceptual_loss's default
        "stages": config["curriculum"]["stages"],
        "style": config["style"],
        "content": config["content"],
        "steps": None,
        "commit": None,
    }
    model_json.write_text(json.dumps(info, indent=2) + "\n")
    print(f"{name}: wrote {model_json} ({size})")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--force", action="store_true", help="replace an existing model.json")
    parser.add_argument("names", nargs="+", metavar="NAME")
    args = parser.parse_args()

    failed = False
    for name in args.names:
        error = write_model_json(name, args.models_dir, args.force)
        if error:
            print(f"{name}: skipped: {error}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
