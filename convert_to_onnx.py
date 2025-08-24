#!/usr/bin/env python3

import torch
import torch.onnx
import os
from style_transfer.img_transformer import Transformer

def convert_to_onnx(pth_path, onnx_path):
    """
    Convert a PyTorch model (.pth file) to ONNX format.
    
    Args:
        pth_path (str): Path to the .pth file
        onnx_path (str): Path where the .onnx file will be saved
    """
    
    # Check if input file exists
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    # Load the model
    print(f"Loading model from {pth_path}...")
    
    # Initialize the model architecture
    model = Transformer()
    
    # Load the state dict
    checkpoint = torch.load(pth_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensor (typical image size for style transfer)
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    print(f"Converting to ONNX format...")
    
    # Export to ONNX
    torch.onnx.export(
        model,                          # model being run
        dummy_input,                    # model input (or a tuple for multiple inputs)
        onnx_path,                      # where to save the model
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=11,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names=['input'],          # the model's input names
        output_names=['output'],        # the model's output names
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},    # variable length axes
                     'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    
    print(f"Successfully converted model to: {onnx_path}")
    
    # Verify the ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed!")
    except ImportError:
        print("Note: Install 'onnx' package to verify the converted model")
    except Exception as e:
        print(f"Warning: ONNX model verification failed: {e}")


if __name__ == "__main__":
    # Hardcoded paths - modify these as needed
    PTH_PATH = "/Users/jackblackburn/code/main/style-transfer/py/models/basic/Ukiyo_e/checkpoints/ckpt_stage1_epoch4.pth"
    ONNX_PATH = "/Users/jackblackburn/code/main/style-transfer/py/models/onnx/ukiyo_basic.onnx"
    
    try:
        convert_to_onnx(PTH_PATH, ONNX_PATH)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
