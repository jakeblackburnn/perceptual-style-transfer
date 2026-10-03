import torch
import torch.nn as nn

from style_transfer.feature_extractors.vgg import get_vgg_model

def gram_matrix(feature_map):
    b, c, h, w = feature_map.size()
    features = feature_map.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)

def total_variation(images):
    # sum of squared differences between vertically and horizontally adjacent
    # pixels, averaged over the batch
    vertical   = ((images[:, :, 1:, :] - images[:, :, :-1, :]) ** 2).sum()
    horizontal = ((images[:, :, :, 1:] - images[:, :, :, :-1]) ** 2).sum()
    return (vertical + horizontal) / images.size(0)

def perceptual_loss(generated_images,
                    content_images,
                    style_images,
                    content_weight=1.0,
                    style_weight=2.5,
                    tv_weight=0.0):
    """Score a generated image against content/style targets via VGG features.

    Decoupled from the generator: the caller runs the model and passes in
    `generated_images`, so this is reusable as a training signal for any
    image-producing model (e.g. a diffusion model's denoised output), not
    just the feed-forward StyleTransferModel.

    Uses the reduction of Johnson et al. 2016: the style term of a layer is
    the squared Frobenius norm of the Gram difference (a SUM over the CxC
    entries, not a mean), scaled by that layer's weight from the preset's
    `style_layer_weights`. The default style_weight of 2.5 is sized for this
    sum reduction. total = content_weight * content + style_weight * style
    + tv_weight * tv, where tv is the total variation of `generated_images`.
    """

    vgg = get_vgg_model()

    content_layer = vgg.preset_config['content_layer']

    style_layers        = vgg.preset_config['style_layers']
    style_layer_weights = vgg.preset_config.get('style_layer_weights') or [1.0] * len(style_layers)
    use_raw_features    = vgg.preset_config.get('use_raw_features', False)

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
            # Vectorized computation of all content-style pairs
            gen_expanded = gen_features.unsqueeze(1)      # [B_content, 1, C, H, W]
            style_expanded = style_features.unsqueeze(0)  # [1, B_style, C, H, W]
            pairwise_losses = ((gen_expanded - style_expanded) ** 2).mean(dim=(-3, -2, -1))  # [B_content, B_style]
            style_loss += style_layer_weights[idx] * pairwise_losses.mean()  # Normalize by total pairs
        else:
            gen_grams = gram_matrix(gen_feats[layer])
            style_grams = gram_matrix(style_feats[layer])
            # Vectorized computation of all content-style gram matrix pairs
            gen_expanded = gen_grams.unsqueeze(1)        # [B_content, 1, C, C]
            style_expanded = style_grams.unsqueeze(0)    # [1, B_style, C, C]
            # squared Frobenius norm: sum (not mean) over the CxC entries
            pairwise_losses = ((gen_expanded - style_expanded) ** 2).sum(dim=(-2, -1))  # [B_content, B_style]
            style_loss += style_layer_weights[idx] * pairwise_losses.mean()  # Normalize by total pairs

    total_loss = content_weight * content_loss + style_weight * style_loss

    if tv_weight != 0.0:
        total_loss = total_loss + tv_weight * total_variation(generated_images)

    return total_loss
