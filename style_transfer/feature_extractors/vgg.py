import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from collections import OrderedDict

# Import layer presets from dedicated config module
from ..config.layer_presets import get_legacy_preset_config

class VGG(nn.Module):
    
    def __init__(self, style_layers, preset_name=None, preset_config=None):
        super().__init__()                
        self.style_layers = style_layers
        self.preset_name = preset_name
        self.preset_config = preset_config

        # load VGG19 and grab its features
        vgg_pretrained = vgg19(weights=VGG19_Weights.DEFAULT).features
        for p in vgg_pretrained.parameters():
            p.requires_grad = False
        self.vgg = vgg_pretrained

    def forward(self, x):
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)

        x = (x - mean) / std # normalized input

        # return only features extracted from specified style layers
        outputs = OrderedDict()
        for name, module in self.vgg._modules.items():
            x = module(x) # pass through next layer
            if name in self.style_layers:
                outputs[name] = x
            # note: style layers includes content layer 
            # content layer is last in the orderedDict

        return outputs

# Global VGG model instance for persistent use
_vgg_model = None

def initialize_vgg(layer_preset='standard', device='cpu', style_specific_presets=None):
    global _vgg_model # persistend vgg model object

    # Get layer configuration from preset (checks style-specific first, then global)
    preset_config = get_legacy_preset_config(layer_preset, style_specific_presets)

    content_layer = preset_config['content_layer']
    style_layers = preset_config['style_layers']

    # Create VGG model with layers from preset
    _vgg_model = VGG(style_layers + [content_layer], layer_preset, preset_config).eval()
    if device is not None:
        _vgg_model = _vgg_model.to(device)

def get_vgg_model():
    if _vgg_model is None:
        raise RuntimeError("VGG model not initialized. Call initialize_vgg() first. idiot.")
    return _vgg_model
