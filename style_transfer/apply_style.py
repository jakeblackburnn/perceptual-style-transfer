import os
import re
import torch
from PIL import Image
from torchvision import transforms
from style_transfer.models import StyleTransferModel
from style_transfer.config import Models

# Configuration
MODEL_NAME = 'high_starry_night'  
USE_FINAL_MODEL_ONLY = True  
MODEL_EXTENSION = '.pth'
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

model_config = Models.get(MODEL_NAME)
model_size = model_config.get('model_size')
model_dir = f"models/{MODEL_NAME}/checkpoints"
content_dir = "images/content/frogs"
output_dir = f"outputs/{MODEL_NAME}"

def setup_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    return device

def natural_key(fname):
    """Sort files by the first integer in their filename."""
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else fname

def create_model(model_size, model_path, device):
    model = StyleTransferModel(size_config=model_size).to(device).eval()
    ckpt_data = torch.load(model_path, map_location=device)
    
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict)
    
    return model

def setup_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def get_output_path(use_final_model, output_dir, img_basename, model_filename=None):
    if use_final_model:
        return output_dir, f"{img_basename}.jpg"
    else:
        img_output_dir = os.path.join(output_dir, img_basename)
        os.makedirs(img_output_dir, exist_ok=True)
        return img_output_dir, f"{model_filename}.jpg"

def process_single_image(content_path, model, device, output_path):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(0, 1)),
        transforms.ToPILImage()
    ])
    
    # Load and preprocess
    content_img = Image.open(content_path).convert('RGB')
    content_tensor = preprocess(content_img).unsqueeze(0).to(device)
    
    # Transform
    with torch.no_grad():
        output_tensor = model(content_tensor)
    
    # Save result
    output_img = postprocess(output_tensor.squeeze(0).cpu())
    output_img.save(output_path)
    return output_path

def main():
    device = setup_device()
    setup_directories(output_dir)
    
    # Gather model files to process
    if USE_FINAL_MODEL_ONLY:
        final_model_path = f"models/{MODEL_NAME}/{MODEL_NAME}{MODEL_EXTENSION}"
        if not os.path.exists(final_model_path):
            raise FileNotFoundError(f"Final model not found at {final_model_path}")
        models_to_process = [(final_model_path, MODEL_NAME)]
    else:
        ckpts = [f for f in os.listdir(model_dir) if f.endswith(MODEL_EXTENSION)]
        ckpts = sorted(ckpts, key=natural_key)
        models_to_process = [(os.path.join(model_dir, ckpt), os.path.splitext(ckpt)[0]) for ckpt in ckpts]
    
    # Get content images
    content_images = [f for f in os.listdir(content_dir) 
                     if os.path.splitext(f.lower())[1] in VALID_IMAGE_EXTENSIONS]
    content_images = sorted(content_images)
    
    print(f"Found {len(content_images)} content images to process")
    
    if USE_FINAL_MODEL_ONLY:
        # Load model once for all images
        model_path, model_filename = models_to_process[0]
        print(f'Loading model: {model_filename}')
        model = create_model(model_size, model_path, device)
        
        # Process each content image with the same model
        for content_img_name in content_images:
            content_img_path = os.path.join(content_dir, content_img_name)
            img_basename = os.path.splitext(content_img_name)[0]
            
            print(f"Processing content image: {content_img_name}")
            
            img_output_dir, out_filename = get_output_path(USE_FINAL_MODEL_ONLY, output_dir, img_basename)
            out_path = os.path.join(img_output_dir, out_filename)
            
            process_single_image(content_img_path, model, device, out_path)
            print(f'  → saved {out_path}')
    else:
        # Checkpoint mode: load model per checkpoint per image
        for content_img_name in content_images:
            content_img_path = os.path.join(content_dir, content_img_name)
            img_basename = os.path.splitext(content_img_name)[0]
            
            print(f"\nProcessing content image: {content_img_name}")
            
            for model_path, model_filename in models_to_process:
                print(f'  Loading model: {model_filename}')
                model = create_model(model_size, model_path, device)
                
                img_output_dir, out_filename = get_output_path(USE_FINAL_MODEL_ONLY, output_dir, img_basename, model_filename)
                out_path = os.path.join(img_output_dir, out_filename)
                
                process_single_image(content_img_path, model, device, out_path)
                print(f'    → saved {out_path}')

if __name__ == "__main__":
    main()
