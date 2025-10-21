import sys
import os
import subprocess
import torch
from pathlib import Path

from style_transfer.config import Models
from style_transfer.train import train_model
from style_transfer.models import StyleTransferModel
from PIL import Image
from torchvision import transforms


def setup_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def train_experiment(experiment_name, model_config, device):
    print(f"\n{'='*60}")
    print(f"TRAINING: {experiment_name}")
    print(f"{'='*60}\n")

    train_model(experiment_name, model_config, device)

    print(f"\n✓ Training complete: {experiment_name}")


def apply_style(experiment_name, model_config, device, content_dir="artifacts/images/content/frogs", output_dir=None):
    print(f"\n{'='*60}")
    print(f"APPLYING STYLE: {experiment_name}")
    print(f"{'='*60}\n")

    # Setup paths
    model_size = model_config.get('model_size', 'medium')
    model_path = f"models/{experiment_name}/{experiment_name}.pth"

    if not os.path.exists(model_path):
        print(f"✗ Error: Model not found at {model_path}")
        print(f"  Make sure training completed successfully.")
        return False
    if output_dir is None:
        output_dir = f"outputs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}")
    model = StyleTransferModel(size_config=model_size).to(device).eval()
    ckpt_data = torch.load(model_path, map_location=device)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage()
    ])

    # Get content images
    content_path = Path(content_dir)
    if not content_path.exists():
        print(f"✗ Error: Content directory not found: {content_dir}")
        return False

    valid_extensions = {'.jpg', '.jpeg', '.png'}
    content_images = [f for f in content_path.iterdir()
                     if f.suffix.lower() in valid_extensions]

    if not content_images:
        print(f"✗ Error: No images found in {content_dir}")
        return False

    content_images = sorted(content_images)
    print(f"Found {len(content_images)} images to process\n")

    # Process each image
    for img_path in content_images:
        print(f"Processing {img_path.name}...", end=" ")

        # Load and preprocess
        content_img = Image.open(img_path).convert('RGB')
        content_tensor = preprocess(content_img).unsqueeze(0).to(device)

        # Transform
        with torch.no_grad():
            output_tensor = model(content_tensor)

        # Save result
        output_img = postprocess(output_tensor.squeeze(0).cpu())
        output_path = Path(output_dir) / f"{img_path.stem}.jpg"
        output_img.save(output_path)

        print(f"✓ saved to {output_path}")

    print(f"\n✓ Applied style to {len(content_images)} images")
    print(f"  Output directory: {output_dir}")

    subprocess.run(["open", output_dir]) # open the folder of stylized images

    return True


def select_experiment_interactive():
    print("\nAvailable Experiments:")
    print("=" * 60)

    # Define style categories (order matters for display)
    style_categories = {
        'kanagawa': [],
        'starry': [],
        'collioure': [],
        'colors1': [],
        'colors2': []
    }

    # Categorize experiments by style
    for exp in Models.keys():
        exp_lower = exp.lower()
        categorized = False
        for style_key in style_categories.keys():
            if style_key in exp_lower:
                style_categories[style_key].append(exp)
                categorized = True
                break

        if not categorized:
            # Fallback: use first word as category
            prefix = exp.split('_')[0]
            if prefix not in style_categories:
                style_categories[prefix] = []
            style_categories[prefix].append(exp)

    idx = 1
    exp_map = {}
    for style, exps in style_categories.items():
        if not exps:  # Skip empty categories
            continue

        print(f"\n{style.upper()}:")
        for exp in sorted(exps):
            config = Models.get(exp)
            model_size = config.get('model_size', 'medium')
            print(f"  {idx:2d}. {exp:30s} [{model_size}]")
            exp_map[idx] = exp
            idx += 1

    print("\n" + "=" * 60)

    # Get user selection
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

    if experiment_name not in Models:
        print(f"✗ Error: Experiment '{experiment_name}' not found")
        return True  # Continue loop

    model_config = Models.get(experiment_name)

    action = select_action()
    if action is None:
        return False  # Exit

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    print(f"Model size: {model_config.get('model_size', 'medium')}")
    print(f"Layer preset: {model_config.get('layer_preset', 'standard')}")

    device = setup_device()

    if action == 1:  # Train only
        train_experiment(experiment_name, model_config, device)
        print(f"\n✓ Training complete: {experiment_name}")

    elif action == 2:  # Apply only
        # Prompt for input directory
        content_dir = input(f"\nEnter input directory (default: artifacts/images/content/frogs): ").strip()
        if not content_dir:
            content_dir = "artifacts/images/content/frogs"
        success = apply_style(experiment_name, model_config, device, content_dir=content_dir)
        if not success:
            print(f"\n✗ Style application failed")

    elif action == 3:  # Train and apply
        train_experiment(experiment_name, model_config, device)
        # Prompt for input directory
        content_dir = input(f"\nEnter input directory (default: artifacts/images/content/frogs): ").strip()
        if not content_dir:
            content_dir = "artifacts/images/content/frogs"
        success = apply_style(experiment_name, model_config, device, content_dir=content_dir)
        if not success:
            print(f"\n✗ Style application failed")

    print(f"\n{'='*60}")
    print(f"✓ EXPERIMENT COMPLETE: {experiment_name}")
    print(f"{'='*60}\n")

    return True  # Continue loop


def main():
    print("\n" + "=" * 60)
    print("PyTorch Style Transfer - Experiment Runner")
    print("=" * 60)

    # Main loop
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
