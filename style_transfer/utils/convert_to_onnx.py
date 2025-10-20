import torch
import torch.onnx
import os
from style_transfer.models import StyleTransferModel
from style_transfer.config import Models

def convert_to_onnx(model_name):
    
    pth_path = f"models/{model_name}/{model_name}.pth"
    onnx_path = f"models/{model_name}/{model_name}.onnx"

    # Check if input file exists
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    # Load the model
    print(f"Loading model from {pth_path}...")
    
    model_size = Models.get(model_name).get('model_size')
    model = StyleTransferModel(size_config=model_size)
    
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
    
    model_name = "mini_kanagawa"

    try:
        convert_to_onnx(model_name)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
