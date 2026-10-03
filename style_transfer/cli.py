"""The `style-transfer` command: list, stylize, train, serve.

Exit codes: 0 success, 2 usage error or missing prerequisite, 1 processing
failure.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from style_transfer.inference import (
    DEFAULT_MODELS_DIR,
    list_models,
    load_named_model,
    select_device,
    stylize_image,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEVICES = ["cpu", "cuda", "mps"]


def _error(message):
    print(f"error: {message}", file=sys.stderr)


def cmd_list(args):
    try:
        models = list_models(args.models_dir)
    except ValueError as e:
        _error(e)
        return 1

    print(f"Installed models ({args.models_dir}):")
    if models:
        for info in models:
            print(f"  {info['name']:32s} {info['model_size']:8s} {info.get('loss', '-')}")
    else:
        print("  none. Train one with `style-transfer train --experiment NAME`, or describe")
        print("  existing weights with `python scripts/write_model_json.py NAME`.")

    from style_transfer.config import Models

    print("\nTraining experiments (style-transfer train --experiment NAME):")
    for name in sorted(Models):
        config = Models[name]
        print(f"  {name:32s} {config.get('model_size', 'medium'):8s} {config.get('layer_preset', 'standard')}")
    return 0


def _plan_outputs(args):
    """Return [(input_path, output_path), ...], or None after printing a usage error."""
    source = Path(args.input)
    if source.is_file():
        if not args.output or args.output_dir:
            _error("an image file needs -o FILE (and not --output-dir)")
            return None
        return [(source, Path(args.output))]
    if source.is_dir():
        if not args.output_dir or args.output:
            _error("a directory needs --output-dir DIR (and not -o)")
            return None
        images = sorted(f for f in source.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        if args.limit is not None:
            images = images[:args.limit]
        if not images:
            _error(f"no jpg/jpeg/png images in {source}")
            return None
        stems = Counter(image.stem for image in images)
        clashes = sorted(image.name for image in images if stems[image.stem] > 1)
        if clashes:
            _error(f"these inputs would write the same output file: {', '.join(clashes)}")
            return None
        return [(image, Path(args.output_dir) / f"{image.stem}.png") for image in images]
    _error(f"input not found: {source}")
    return None


def cmd_stylize(args):
    plan = _plan_outputs(args)
    if plan is None:
        return 2

    if not args.overwrite:
        existing = [str(output) for _, output in plan if output.exists()]
        if existing:
            _error(f"refusing to overwrite (pass --overwrite): {', '.join(existing)}")
            return 2

    device = args.device or select_device()
    try:
        model = load_named_model(args.model, args.models_dir, device)
    except FileNotFoundError as e:
        _error(e)
        return 2
    except ValueError as e:
        _error(e)
        return 1

    failed = 0
    for source, output in plan:
        try:
            with Image.open(source) as img:
                result = stylize_image(img, model, device, max_side=args.max_side)
            output.parent.mkdir(parents=True, exist_ok=True)
            result.save(output)
        except (OSError, ValueError) as e:
            _error(f"{source}: {e}")
            failed += 1
            continue
        print(f"{source} -> {output} ({result.width}x{result.height})")
    return 1 if failed else 0


def cmd_train(args):
    from style_transfer.config import Models

    if args.experiment not in Models:
        _error(f"unknown experiment '{args.experiment}'. Valid names: {', '.join(sorted(Models))}")
        return 2

    import torch

    from style_transfer.train import train_model

    device = torch.device(args.device) if args.device else select_device()
    weights = train_model(args.experiment, Models[args.experiment], device, models_dir=args.models_dir)
    print(f"Saved {weights}")
    print(f"Try it: style-transfer --models-dir {args.models_dir} stylize PHOTO.jpg "
          f"--model {args.experiment} -o out.png")
    return 0


def cmd_serve(args):
    try:
        from style_transfer_web.app import run
    except ImportError as e:
        _error(f"the web UI is not installed ({e}). Install it with: pip install -e '.[ui]'")
        return 2
    run(args.models_dir, port=args.port, device=args.device)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="style-transfer", description="Perceptual-loss style transfer.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="directory of trained models (default: %(default)s)")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="show installed models and training experiments")
    list_parser.set_defaults(run=cmd_list)

    stylize = commands.add_parser("stylize", help="stylize an image or a directory of images")
    stylize.add_argument("input", metavar="INPUT", help="an image file, or a directory of jpg/jpeg/png files")
    stylize.add_argument("--model", required=True, metavar="NAME", help="installed model name (see `list`)")
    stylize.add_argument("-o", "--output", metavar="FILE",
                         help="output file for a single image; its extension sets the format")
    stylize.add_argument("--output-dir", metavar="DIR", help="output directory for a directory input (<stem>.png)")
    stylize.add_argument("--max-side", type=int, metavar="N", help="downsize so the longer side is at most N px")
    stylize.add_argument("--device", choices=DEVICES, help="default: best available")
    stylize.add_argument("--limit", type=int, metavar="N", help="process at most N images of a directory")
    stylize.add_argument("--overwrite", action="store_true", help="replace existing output files")
    stylize.set_defaults(run=cmd_stylize)

    train = commands.add_parser("train", help="train a registered experiment")
    train.add_argument("--experiment", required=True, metavar="NAME", help="experiment name (see `list`)")
    train.add_argument("--device", choices=DEVICES, help="default: best available")
    train.set_defaults(run=cmd_train)

    serve = commands.add_parser("serve", help="run the local web UI on 127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", choices=DEVICES, help="default: best available")
    serve.set_defaults(run=cmd_serve)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
