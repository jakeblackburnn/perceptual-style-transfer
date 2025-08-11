import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from collections import OrderedDict

# pretrained classification model
class VGG(nn.Module):
    
    def __init__(self, style_layers):

        super().__init__()                
        self.style_layers = style_layers

        # load VGG19 and grab its features
        vgg_pretrained = vgg19(weights=VGG19_Weights.DEFAULT).features
        for p in vgg_pretrained.parameters():
            p.requires_grad = False
        self.vgg = vgg_pretrained


    def forward(self, x):
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        outputs = OrderedDict()
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.style_layers:
                outputs[name] = x
        return outputs



# build perception model w/ given layers
style_layers = ['0', '5', '10', '19', '28'] 
content_layer = '21'
vgg = VGG(style_layers + [content_layer]).eval() 

# allow other code to move vgg to other devices
def set_vgg_device(device: torch.device):
    vgg.to(device)

# gram matrix for distilling style info from current layer features
def gram_matrix(feature_map):
    b, c, h, w = feature_map.size()
    features = feature_map.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)

# perceptual loss defined by vgg model features
def vgg_perceptual_loss(model, content_images, style_images,
                        content_layer='21',                       
                        style_layers=['0', '5', '10', '19', '28'],
                        content_weight=1.0,
                        style_weight=1e5):
    """
    VGG perceptual loss with explicit all-pairs style comparison.
    
    For B_content content images and B_style style images:
    - Generates B_content stylized images (one per content image)
    - Content loss: each generated vs its corresponding original content
    - Style loss: each generated vs EVERY style image individually
    - Total style comparisons: B_content × B_style pairings
    
    Example: 2 content + 3 style images = 2 content comparisons + 6 style comparisons
    """
    # content_images: [B_content, 3, H, W] - flexible number of content images
    # style_images: [B_style, 3, H, W] - flexible number of style images
    
    B_content = content_images.size(0)
    B_style = style_images.size(0)
    
    # Generate stylized images
    generated_images = model(content_images)  # [B_content, 3, H, W]

    # Extract VGG features for generated and content images
    gen_feats = vgg(generated_images)        # Dict: layer_name -> [B_content, C, H, W]
    content_feats = vgg(content_images)      # Dict: layer_name -> [B_content, C, H, W]
    
    # Extract VGG features for style images
    style_feats = vgg(style_images)  # Dict: layer_name -> [B_style, C, H, W]

    # Content loss: each generated image vs its corresponding original content image
    content_loss = nn.MSELoss()(gen_feats[content_layer], content_feats[content_layer])

    # Style loss: explicit all-pairs comparison
    # Each generated image compared against every style image individually
    style_loss = 0.0
    for layer in style_layers:
        # Compute gram matrices for this layer
        gen_grams = gram_matrix(gen_feats[layer])    # [B_content, feat_size, feat_size]
        style_grams = gram_matrix(style_feats[layer]) # [B_style, feat_size, feat_size]
        
        # Explicit double loop: all content-style pairings
        for content_idx in range(B_content):
            for style_idx in range(B_style):
                style_loss += nn.MSELoss()(gen_grams[content_idx], style_grams[style_idx])

    return content_weight * content_loss + style_weight * style_loss
