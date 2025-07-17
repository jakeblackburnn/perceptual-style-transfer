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
def perceptual_loss(model, batch,
                        content_layer='21',                       
                        style_layers=['0', '5', '10', '19', '28'],
                        content_weight=1.0,
                        style_weight=1e5):

    content_images, style_images = batch
    generated_images = model(content_images)

    # Extract features from VGG
    gen_feats     = vgg(generated_images)
    content_feats = vgg(content_images)
    style_feats   = vgg(style_images)

    # Content loss = mse( generated content feats - original content feats )
    content_loss = nn.MSELoss()(gen_feats[content_layer], content_feats[content_layer])

    # Style loss = sum( mse( generated gram matrix - original gram matrix ) )
    style_loss = 0.0
    for layer in style_layers:
        target_gram = gram_matrix(style_feats[layer])
        gen_gram    = gram_matrix(gen_feats[layer])

        style_loss += nn.MSELoss()(gen_gram, target_gram)

    return content_weight * content_loss + style_weight * style_loss
