import torch
import torch.nn as nn

from style_transfer.feature_extractors.vgg import get_vgg_model

# gram matrix for distilling style info from current layer features
def gram_matrix(feature_map):
    b, c, h, w = feature_map.size()
    features = feature_map.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)

# perceptual loss defined by vgg model features
def vgg_perceptual_loss(model, content_images, style_images,
                        content_weight=1.0,
                        style_weight=1e5):
    
    # Use the pre-initialized global VGG model
    vgg = get_vgg_model()
    
    # Get layer configuration directly from the VGG model
    content_layer = vgg.preset_config['content_layer']
    style_layers = vgg.preset_config['style_layers']

    style_layer_weights = vgg.preset_config.get('style_layer_weights', None)
    if style_layer_weights == None:
        style_layer_weights = [1.0] * len(style_layers)

    # batch sizes
    B_content = content_images.size(0)
    B_style = style_images.size(0)
    
    # forward pass 
    generated_images = model(content_images)  # [B_content, 3, H, W]

    gen_feats = vgg(generated_images)        # Dict: layer_name -> [B_content, C, H, W]
    content_feats = vgg(content_images)      # Dict: layer_name -> [B_content, C, H, W]
    style_feats = vgg(style_images)          # Dict: layer_name -> [B_style, C, H, W]

    content_loss = nn.MSELoss()(gen_feats[content_layer], content_feats[content_layer])

    style_loss = 0.0
    for idx, layer in enumerate(style_layers):

        gen_grams = gram_matrix(gen_feats[layer])    # [B_content, feat_size, feat_size]
        style_grams = gram_matrix(style_feats[layer]) # [B_style, feat_size, feat_size]
        
        # sum style loss over all content-style pairings
        for content_idx in range(B_content):
            for style_idx in range(B_style):
                style_loss += style_layer_weights[idx] * nn.MSELoss()(gen_grams[content_idx], style_grams[style_idx])

    return content_weight * content_loss + style_weight * style_loss
