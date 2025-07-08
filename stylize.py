import torch
from PIL import Image
from torchvision import transforms

from models.img_transformer import TransformerNet

# Hardcoded paths and device
MODEL_PATH = 'models/transformer_final.pth'
CONTENT_IMG = 'images/content/frogs/bluepoisondartfrog2.jpeg'
OUTPUT_IMG = 'output/stylized.jpg'
DEVICE = torch.device('cpu')  # CPU only

# Load the trained transformer network
transformer = TransformerNet()
state_dict = torch.load(MODEL_PATH, map_location='cpu')
transformer.load_state_dict(state_dict)
transformer.to(DEVICE).eval()

# Preprocess content image
image = Image.open(CONTENT_IMG).convert('RGB')
preprocess = transforms.Compose([
    transforms.ToTensor(),             # [0,1]
    transforms.Lambda(lambda x: x * 255)  # scale to [0,255]
])
content_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# Stylize
with torch.no_grad():
    output_tensor = transformer(content_tensor).cpu()

# Post-process and save
# TransformerNet outputs tensors in [0,1]
postprocess = transforms.Compose([
    transforms.Lambda(lambda x: x.clamp(0, 1)),  # ensure range [0,1]
    transforms.ToPILImage()
])

# Ensure output directory exists
import os
os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)

output_image = postprocess(output_tensor.squeeze(0))
output_image.save(OUTPUT_IMG)
print(f"Stylized image saved to {OUTPUT_IMG}")

