# VGG Layer Presets Configuration
# Enhanced presets with explicit feature extractor specification and improved structure

# Standard layer presets with enhanced structure
LAYER_PRESETS = {
    'standard': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10', '19', '28'],  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        'content_layer': '21',  # conv4_2
        'style_layer_weights': None,
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'standard_weighted': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10', '19', '28'],  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        'content_layer': '21',  # conv4_2
        'style_layer_weights': [0.5, 1.0, 2.0, 2.5, 3.0],
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'shallow': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '2', '5', '7', '10'],  # early conv layers for fine texture
        'content_layer': '21',  
        'style_layer_weights': None,
        'use_raw_features': False  # use gram matrices for style transfer
    },

    'deep': {
        'feature_extractor': 'vgg19',
        'style_layers': ['10', '19', '28'],  # later conv layers for semantic style
        'content_layer': '28',  # conv5_1 for high-level content
        'style_layer_weights': None,
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
    }
}

def get_layer_preset(preset_name, style_specific_presets=None):
    """
    Get layer preset configuration, checking style-specific presets first.

    Args:
        preset_name (str): Name of the preset to retrieve
        style_specific_presets (dict, optional): Style-specific presets to check first

    Returns:
        dict: Layer preset configuration

    Raises:
        KeyError: If preset not found in either style-specific or global presets
    """
    # Check style-specific presets first
    if style_specific_presets and preset_name in style_specific_presets:
        return style_specific_presets[preset_name]

    # Fall back to global presets
    if preset_name in LAYER_PRESETS:
        return LAYER_PRESETS[preset_name]

    # Preset not found
    available_presets = list(LAYER_PRESETS.keys())
    if style_specific_presets:
        available_presets.extend(list(style_specific_presets.keys()))

    raise KeyError(f"Layer preset '{preset_name}' not found. Available presets: {available_presets}")

# Legacy compatibility - maps new format back to old VGG format
def get_legacy_preset_config(preset_name, style_specific_presets=None):
    """
    Get preset config in the legacy format expected by VGG module.
    This maintains backward compatibility during transition.
    """
    preset = get_layer_preset(preset_name, style_specific_presets)

    # Convert to legacy format (remove feature_extractor field)
    legacy_config = {
        'style_layers': preset['style_layers'],
        'content_layer': preset['content_layer'],
        'use_raw_features': preset['use_raw_features']
    }

    # Add style_layer_weights only if present
    if preset['style_layer_weights'] is not None:
        legacy_config['style_layer_weights'] = preset['style_layer_weights']

    return legacy_config
