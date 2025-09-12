import torch
import torch.nn as nn

from style_transfer.feature_extractors.vgg import get_vgg_model

def gram_matrix(feature_map):
    b, c, h, w = feature_map.size()
    features = feature_map.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)

def vgg_perceptual_loss(model, 
                        content_images, 
                        style_images,
                        content_weight=1.0,
                        style_weight=1e5):
    
    vgg = get_vgg_model()
    
    content_layer = vgg.preset_config['content_layer']

    style_layers        = vgg.preset_config['style_layers']
    style_layer_weights = vgg.preset_config.get( 'style_layer_weights', [1.0] * len(style_layers) )
    use_raw_features    = vgg.preset_config.get('use_raw_features', False)

    b_content = content_images.size(0)
    b_style = style_images.size(0)
    
    generated_images = model(content_images)  # [B_content, 3, H, W]

    # features extracted from vgg
    gen_feats = vgg(generated_images)        # Dict: layer_name -> [B_content, C, H, W]
    content_feats = vgg(content_images)      # Dict: layer_name -> [B_content, C, H, W]
    style_feats = vgg(style_images)          # Dict: layer_name -> [B_style, C, H, W]

    content_loss = nn.MSELoss()(gen_feats[content_layer], content_feats[content_layer])

    style_loss = 0.0
    for idx, layer in enumerate(style_layers):
        
        if use_raw_features:
            gen_features = gen_feats[layer]      # [B_content, C, H, W]
            style_features = style_feats[layer]  # [B_style, C, H, W]
            for content_idx in range(b_content):
                for style_idx in range(b_style):
                    style_loss += nn.MSELoss()(gen_features[content_idx], style_features[style_idx])
        else:
            gen_grams = gram_matrix(gen_feats[layer])
            style_grams = gram_matrix(style_feats[layer])

            for content_idx in range(b_content):
                for style_idx in range(b_style):
                    style_loss += nn.MSELoss()(gen_grams[content_idx], style_grams[style_idx])

    return content_weight * content_loss + style_weight * style_loss
