# Neural Style Transfer

A PyTorch implementation of neural style transfer using VGG-based perceptual loss with configurable training curricula and multiple model architectures. Supports a fully configurable build pipeline with automatic metrics logging and an intuitive inference script. Also includes scripts for visualizing VGG activations, informing the fine-tuning of the perceptual loss calculation, as well as a conversion script for saving pytorch models to ONNX format. 

## Features

- **Multiple Model Architectures**: Small, medium, and large sized transformers
- **Configurable Training Curricula**: Progressive training with adjustable resolution, batch size, epochs, and learning rates
- **VGG Perceptual Loss**: Fine-tuned content and style loss computation using pre-trained VGG network
- **Flexible Dataset Support**: Handles arbitrary content and style image collections
- **GPU Acceleration**: Support for CUDA and MPS (Apple Silicon) backends
- **Model Export**: Convert trained PyTorch models to ONNX format
- **Comprehensive Metrics**: Training loss tracking and checkpoint management

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd style-transfer/py
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision pillow
   ```

## Usage

### Training a Model

Configure your model's training settings in `style_transfer/config.py`, then specify the model name in build.py. initiate training with:

```bash
python build.py
```

### Applying Style Transfer

1. Place images somewhere in `images/`.
2. Set the model name and input image directory in `apply_style.py`
3. Run inference:
   ```bash
   python apply_style.py
   ```

### Converting to ONNX

specify the model name to convert. 

```bash
python convert_to_onnx.py
```

## Project Structure

- `style_transfer/` - Core style transfer implementation
  - `train.py` - Training loop and epoch management
  - `loss.py` - VGG perceptual loss functions
  - `dataset.py` - Image dataset handling
  - `config.py` - Model and training configurations
- `utils/` - Utility functions for metrics, activation extraction, and visualization
- `models/` - Trained models, training data / metrics, and checkpoints 
- `images/` - Content and style image collections
- `outputs/` - Generated stylized images

## Technologies Used

- **Framework**: PyTorch, torchvision
- **Image Processing**: PIL (Pillow)
- **Model Export**: ONNX
- **Compute**: CUDA, MPS (Apple Silicon)

## Training Curriculum

The system supports progressive training curricula with configurable:
- Image resolution 
- Batch processing  
- Learning rates and epochs per stage
- Content/style loss weights

## License

This project is licensed under the [MIT License](LICENSE).
