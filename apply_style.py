import os
import re
import torch
from PIL import Image
from torchvision import transforms
from style_transfer.img_transformer import Transformer

# ——— Configuration ———

curriculum = 'basic_small_batch'

# Directory containing .pth checkpoints
model_dir   = f"models/{curriculum}/checkpoints"
# Path to content image
content_img = f"images/content/frogs/bluepoisondartfrog2.jpeg"
# Directory where outputs will go
output_dir  = f"models/{curriculum}/output"

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

# Load and preprocess the content image once
content_img   = Image.open(content_img).convert('RGB')
content_tensor = preprocess(content_img).unsqueeze(0).to(device)

# Iterate through each checkpoint, stylize and save
for ckpt in ckpts:
    model_path = os.path.join(model_dir, ckpt)
    print(f'Loading checkpoint: {model_path}')
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
    # mirror the checkpoint name for the output filename
    name = os.path.splitext(ckpt)[0] + '.jpg'
    out_path = os.path.join(output_dir, name)
    output_img.save(out_path)
    print(f'  → saved {out_path}')
