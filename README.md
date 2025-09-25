# Perceptual Loss Style Transfer

Professional PyTorch neural style transfer with experiment-based configuration, progressive training curricula, and interactive visualization.

<div>
<img src="examples/frog4.jpg" width="300">
<p>--></p>
<img src="examples/frog5.jpg" width="300">
</div>

## Features

- **Multiple Model Architectures**: Small, medium, and big ResNet-inspired transformers
- **Experiment-Based Configuration**: Pre-configured experiments combining models, datasets, and training curricula
- **Advanced VGG Layer Presets**: Standard, weighted, shallow, deep, and feature blender configurations
- **Multi-Stage Training Curricula**: Progressive training with configurable hyperparameters per stage
- **Interactive Visualization**: Web-based VGG activation explorer with Dash
- **Multi-GPU Support**: CUDA, MPS (Apple Silicon), and CPU with auto-detection
- **ONNX Export**: Deploy models in ONNX format

## Requirements

- Python 3.8+
- PyTorch (CPU/CUDA/MPS support)
- 4GB+ GPU memory recommended for training

## Installation

```bash
git clone <repository-url>
cd perceptual-style-transfer
pip install torch torchvision pillow dash plotly
```

## Usage

### Quick Start

1. **Train a model**: Edit `MODEL_NAME` in `build.py` to select an experiment:
   ```bash
   # Edit line 8: MODEL_NAME = 'kanagawa'  # or 'port_of_collioure', 'big_kanagawa', etc.
   python build.py
   ```

2. **Apply style transfer**: Set model and image paths in `apply_style.py` and run:
   ```bash
   python apply_style.py
   ```

3. **Visualize VGG activations**: Launch interactive web interface:
   ```bash
   python visualize.py
   ```

4. **Export to ONNX**: Convert trained models for deployment:
   ```bash
   python convert_to_onnx.py
   ```

> **Training Time**: ~10-30 minutes per experiment (depends on hardware and curriculum)
>
> **Output**: Trained models saved to `models/{experiment_name}/`, stylized images in `outputs/{experiment_name}/`

## Project Structure

- `style_transfer/` - Core implementation
  - `config/` - Modular configuration system
    - `styles/` - Style-specific experiment definitions
    - `curricula.py` - Training schedules and hyperparameters
    - `layer_presets.py` - VGG layer configurations
  - `feature_extractors/` - VGG and other feature extraction modules
  - `architectures/` - Model architecture components
- `utils/` - Metrics and visualization utilities
- `models/` - Trained models and checkpoints
- `images/` - Content and style image collections
- `outputs/` - Generated stylized images

## Configuration

### Pre-Configured Experiments

Select from optimized experiment combinations:

| Experiment | Model Size | Style Focus | Training Intensity |
|------------|------------|-------------|-------------------|
| `kanagawa` | Medium | Balanced | Standard |
| `high_kanagawa` | Medium | High style | Enhanced |
| `big_kanagawa` | Large | Balanced | Standard |
| `port_of_collioure` | Medium | Fauve style | Enhanced |
| `shallow_kanagawa` | Medium | Texture | Standard |
| `deep_kanagawa` | Medium | Semantic | Standard |

### Advanced Configuration

**VGG Layer Presets**: `standard`, `standard_weighted`, `shallow`, `deep`, `deep_weighted`, `feature_blender`

**Training Curricula**: `standard` (6 epochs), `standard_high_style`, `standard_mega_style`, `quick` (4 epochs), `dry_run` (2 epochs)

**Model Architectures**: Small (32 filters, 2 residual blocks), Medium (32 filters, 5 blocks), Big (64 filters, 9 blocks)

**Technologies**: PyTorch, Dash/Plotly, ONNX, Multi-GPU (CUDA/MPS/CPU)

## License

This project is licensed under the [MIT License](LICENSE).
