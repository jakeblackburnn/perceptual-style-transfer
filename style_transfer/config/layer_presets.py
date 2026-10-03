# VGG Layer Presets Configuration
# Enhanced presets with explicit feature extractor specification and improved structure

# Standard layer presets with enhanced structure
LAYER_PRESETS = {
    'standard': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10', '19', '28'],  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        'content_layer': '21',  # conv4_2
        'style_layer_weights': [1.0, 1.0, 1.0, 1.0, 1.0],
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'standard_weighted': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10', '19', '28'],  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        'content_layer': '21',  # conv4_2
        'style_layer_weights': [0.5, 1.0, 2.0, 2.5, 3.0],
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'standard_x_shallow': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10', '19', '28'],  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        'content_layer': '21',  # conv4_2
        'style_layer_weights': [3.0, 2.0, 1.5, 1.0, 0.5],
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'shallow': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '2', '5', '7', '10'],  # early conv layers for fine texture
        'content_layer': '21',  
        'style_layer_weights': [1.0, 1.0, 1.0, 1.0, 1.0],
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'deep': {
        'feature_extractor': 'vgg19',
        'style_layers': ['10', '19', '28'],  # later conv layers for semantic style
        'content_layer': '28',  # conv5_1 for high-level content
        'style_layer_weights': [1.0, 1.0, 1.0],
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'deep_weighted': {
        'feature_extractor': 'vgg19',
        'style_layers': ['10', '19', '28'],  # later conv layers for semantic style
        'content_layer': '21',  # conv5_1 for high-level content
        'style_layer_weights': [1.5, 1.0, 0.5], 
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'feature_blender': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10', '19', '28'],
        'content_layer': '21',
        'style_layer_weights': [0.2, 0.2, 0.2, 0.2, 0.2],  # Fixed: was strings, now floats
        'use_raw_features': True
    },

    'kanagawa_optimized': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10'],  # Fewer layers for bold, simplified style
        'content_layer': '21',  # Keep standard content layer
        'style_layer_weights': [0.3, 0.4, 0.3],  # Custom weights for kanagawa
        'use_raw_features': False
    }
}

def get_layer_preset(preset_name):
    """Get the layer preset configuration for `preset_name`, or raise KeyError."""
    if preset_name not in LAYER_PRESETS:
        raise KeyError(f"Layer preset '{preset_name}' not found. Available presets: {list(LAYER_PRESETS.keys())}")
    return LAYER_PRESETS[preset_name]
