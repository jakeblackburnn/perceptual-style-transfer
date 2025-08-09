import os
import re
import torch
from PIL import Image
from torchvision import transforms
from style_transfer.img_transformer import Transformer

# ——— Configuration ———

curriculum = 'standard'

# Directory containing .pth checkpoints
model_dir   = f"models/{curriculum}/checkpoints"
# Directory containing content images to process
content_dir = "images/content/frogs"
# Directory where outputs will go
output_dir  = f"models/{curriculum}/output"

# Valid image extensions
valid_extensions = {'.jpg', '.jpeg', '.png'}

# Device setup
print("PyTorch version:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ——— Helpers ———
def natural_key(fname):
    """
    Sort files by the first integer in their filename.
    Falls back to lexicographic if no number is found.
    """
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else fname

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Gather and sort all checkpoint files
ckpts = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
ckpts = sorted(ckpts, key=natural_key)

# Pre and Post processing transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)   # scale [0,1]→[0,255]
])
postprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.clamp(0, 1)),  # ensure within [0,1]
    transforms.ToPILImage()
])

# Get all content images with valid extensions
content_images = [f for f in os.listdir(content_dir) 
                 if os.path.splitext(f.lower())[1] in valid_extensions]
content_images = sorted(content_images)

print(f"Found {len(content_images)} content images to process")

# Process each content image
for content_img_name in content_images:
    content_img_path = os.path.join(content_dir, content_img_name)
    img_basename = os.path.splitext(content_img_name)[0]
    
    print(f"\nProcessing content image: {content_img_name}")
    
    # Create output directory for this content image (named after the image)
    img_output_dir = os.path.join(output_dir, img_basename)
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Load and preprocess the content image
    content_img = Image.open(content_img_path).convert('RGB')
    content_tensor = preprocess(content_img).unsqueeze(0).to(device)
    
    # Process with each checkpoint
    for ckpt in ckpts:
        model_path = os.path.join(model_dir, ckpt)
        print(f'  Loading checkpoint: {ckpt}')
        
        # Initialize network and load weights
        transformer = Transformer().to(device).eval()
        ckpt_dat = torch.load(model_path, map_location=device)
        state_dict = ckpt_dat["model_state_dict"]
        transformer.load_state_dict(state_dict)

        # Stylize
        with torch.no_grad():
            output_tensor = transformer(content_tensor)

        # Convert to PIL and save
        output_img = postprocess(output_tensor.squeeze(0).cpu())
        # Save with checkpoint name in the image-specific folder
        ckpt_name = os.path.splitext(ckpt)[0]
        out_filename = f"{ckpt_name}.jpg"
        out_path = os.path.join(img_output_dir, out_filename)
        output_img.save(out_path)
        print(f'    → saved {out_path}')
