import os
import sys
import subprocess
from pathlib import Path

from PIL import Image

from style_transfer.inference import select_device

from diffuser.config import DiffusionExperiments
from diffuser.train import train_model
from diffuser.inference import load_model, stylize_image


def find_checkpoint(experiment_name):
    """diffuser.train saves to models/<name>/<name>.pth."""
    path = Path("models") / experiment_name / f"{experiment_name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return path


def train_experiment(experiment_name, cfg, device):
    print(f"\n{'='*60}")
    print(f"TRAINING: {experiment_name}")
    print(f"{'='*60}\n")

    train_model(experiment_name, cfg, device)

    print(f"\n✓ Training complete: {experiment_name}")


def apply_style(experiment_name, cfg, device, content_dir="artifacts/images/content/frogs", output_dir=None):
    print(f"\n{'='*60}")
    print(f"APPLYING STYLE: {experiment_name}")
    print(f"{'='*60}\n")

    try:
        model_path = find_checkpoint(experiment_name)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"  Make sure training completed successfully.")
        return False

    if output_dir is None:
        output_dir = f"outputs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}")
    model = load_model(model_path, cfg, device)

    content_path = Path(content_dir)
    if not content_path.exists():
        print(f"✗ Error: Content directory not found: {content_dir}")
        return False

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    content_images = sorted(f for f in content_path.iterdir() if f.suffix.lower() in valid_extensions)

    if not content_images:
        print(f"✗ Error: No images found in {content_dir}")
        return False

    print(f"Found {len(content_images)} images to process\n")

    for img_path in content_images:
        print(f"Processing {img_path.name}...", end=" ")

        content_img = Image.open(img_path).convert('RGB')
        output_img = stylize_image(content_img, model, cfg, device)
        output_path = Path(output_dir) / f"{img_path.stem}.jpg"
        output_img.save(output_path)

        print(f"✓ saved to {output_path}")

    print(f"\n✓ Applied style to {len(content_images)} images")
    print(f"  Output directory: {output_dir}")

    subprocess.run(["open", output_dir])  # open the folder of stylized images

    return True


def select_experiment_interactive():
    print("\nAvailable Diffusion Experiments:")
    print("=" * 60)

    names = sorted(DiffusionExperiments.keys())
    exp_map = {}
    for idx, name in enumerate(names, start=1):
        cfg = DiffusionExperiments[name]
        print(f"  {idx:2d}. {name:30s} [image_size={cfg.image_size}, epochs={cfg.epochs}]")
        exp_map[idx] = name

    print("\n" + "=" * 60)

    while True:
        try:
            choice = input(f"\nSelect experiment (1-{len(exp_map)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Exiting.")
                sys.exit(0)

            choice_num = int(choice)
            if choice_num in exp_map:
                return exp_map[choice_num]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(exp_map)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)


def select_action():
    print("\nWhat would you like to do?")
    print("=" * 60)
    print("  1. Train only")
    print("  2. Apply style only")
    print("  3. Train and apply")
    print("=" * 60)

    while True:
        try:
            choice = input("\nSelect action (1-3) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None

            choice_num = int(choice)
            if choice_num in [1, 2, 3]:
                return choice_num
            else:
                print("Invalid choice. Please enter 1, 2, or 3")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\nExiting.")
            return None


def run_single_experiment():
    experiment_name = select_experiment_interactive()
    if experiment_name is None:
        return False

    if experiment_name not in DiffusionExperiments:
        print(f"✗ Error: Experiment '{experiment_name}' not found")
        return True  # Continue loop

    cfg = DiffusionExperiments[experiment_name]

    action = select_action()
    if action is None:
        return False  # Exit

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Image size: {cfg.image_size}")
    print(f"Epochs: {cfg.epochs}")

    device = select_device()
    print(f"Using device: {device}")

    if action == 1:  # Train only
        train_experiment(experiment_name, cfg, device)

    elif action == 2:  # Apply only
        content_dir = input("\nEnter input directory (default: artifacts/images/content/frogs): ").strip()
        if not content_dir:
            content_dir = "artifacts/images/content/frogs"
        success = apply_style(experiment_name, cfg, device, content_dir=content_dir)
        if not success:
            print(f"\n✗ Style application failed")

    elif action == 3:  # Train and apply
        train_experiment(experiment_name, cfg, device)
        content_dir = input("\nEnter input directory (default: artifacts/images/content/frogs): ").strip()
        if not content_dir:
            content_dir = "artifacts/images/content/frogs"
        success = apply_style(experiment_name, cfg, device, content_dir=content_dir)
        if not success:
            print(f"\n✗ Style application failed")

    print(f"\n{'='*60}")
    print(f"✓ EXPERIMENT COMPLETE: {experiment_name}")
    print(f"{'='*60}\n")

    return True  # Continue loop


def main():
    print("\n" + "=" * 60)
    print("PyTorch Diffusion Style Transfer - Experiment Runner")
    print("=" * 60)

    while True:
        try:
            should_continue = run_single_experiment()
            if not should_continue:
                break

            print("\n" + "=" * 60)
            response = input("Run another experiment? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                break

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break

    print("\nGoodbye!")
    os._exit(0)  # Force exit to kill any hanging threads


if __name__ == '__main__':
    main()
