"""Single source of truth for running a trained StyleTransferModel on an image.

Preprocessing here must stay in lockstep with the training pipeline in
dataset.py: both use plain ToTensor() (float, [0, 1] range). Do not scale by
255 before the forward pass -- models.py's final activation is a tanh
rescaled to [0, 1], trained end-to-end against [0, 1]-range inputs.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from style_transfer.models import StyleTransferModel

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

postprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.clamp(0, 1)),
    transforms.ToPILImage(),
])


def load_model(model_path, model_size, device="cpu"):
    """Load a trained StyleTransferModel from a checkpoint file."""
    model = StyleTransferModel(size_config=model_size).to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


def stylize_image(content_img: Image.Image, model: StyleTransferModel, device="cpu") -> Image.Image:
    """Run style transfer on a PIL image and return the stylized PIL image."""
    content_tensor = preprocess(content_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(content_tensor)
    return postprocess(output_tensor.squeeze(0).cpu())


def select_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoint(experiment_name):
    """Locate a trained checkpoint for an experiment name.

    Checkpoints archived under artifacts/models/ take precedence over
    models/, which is train.py's default (and usually gitignored/transient)
    output location.
    """
    for base in (Path("artifacts/models"), Path("models")):
        candidate = base / experiment_name / f"{experiment_name}.pth"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint found for experiment '{experiment_name}' under "
        f"artifacts/models/ or models/"
    )


def _collect_images(content_path, limit=None):
    if content_path.is_file():
        images = [content_path]
    else:
        valid_extensions = {".jpg", ".jpeg", ".png"}
        images = sorted(f for f in content_path.iterdir() if f.suffix.lower() in valid_extensions)
    return images[:limit] if limit is not None else images


def _cli():
    import argparse

    from style_transfer.config import Models

    parser = argparse.ArgumentParser(
        description="Run small-scale style transfer inference for quick manual testing."
    )
    parser.add_argument("--experiment", help="Experiment name from style_transfer.config.Models "
                                              "(looks up model size and checkpoint path automatically).")
    parser.add_argument("--model", help="Explicit path to a .pth checkpoint (overrides --experiment lookup).")
    parser.add_argument("--size", choices=["small", "medium", "big"],
                         help="Model size preset. Required with --model; inferred from --experiment otherwise.")
    parser.add_argument("--content", required=True,
                         help="Path to a single content image or a directory of images.")
    parser.add_argument("--output", help="Output directory (default: outputs/quick_test/<name>).")
    parser.add_argument("--limit", type=int, help="Process at most this many images from a directory.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Override device auto-detection.")
    args = parser.parse_args()

    if not args.model and not args.experiment:
        parser.error("provide --experiment or --model")

    if args.model:
        if not args.size:
            parser.error("--size is required when using --model")
        model_path = Path(args.model)
        model_size = args.size
        name = model_path.stem
    else:
        if args.experiment not in Models:
            parser.error(f"unknown experiment '{args.experiment}' (see style_transfer/config for valid names)")
        model_path = find_checkpoint(args.experiment)
        model_size = args.size or Models[args.experiment]["model_size"]
        name = args.experiment

    device = torch.device(args.device) if args.device else select_device()
    print(f"Using device: {device}")

    content_path = Path(args.content)
    images = _collect_images(content_path, args.limit)
    if not images:
        parser.error(f"no images found at {content_path}")

    output_dir = Path(args.output) if args.output else Path("outputs/quick_test") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} (size={model_size})")
    model = load_model(model_path, model_size, device)

    for img_path in images:
        content_img = Image.open(img_path).convert("RGB")
        output_img = stylize_image(content_img, model, device)
        out_path = output_dir / f"{img_path.stem}.jpg"
        output_img.save(out_path)
        print(f"  {img_path.name} -> {out_path}")

    print(f"Done. {len(images)} image(s) written to {output_dir}")


if __name__ == "__main__":
    _cli()
