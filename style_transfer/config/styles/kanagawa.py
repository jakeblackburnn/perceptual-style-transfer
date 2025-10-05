# Kanagawa Style Configuration
# All kanagawa-related experiments, datasets, and configurations

# Dependencies will be injected by auto-discovery system

# Kanagawa-specific layer presets (optional - can also use global presets)
KANAGAWA_LAYER_PRESETS = {
    'kanagawa_optimized': {
        'feature_extractor': 'vgg19',
        'style_layers': ['0', '5', '10'],  # Fewer layers for bold, simplified style
        'content_layer': '21',  # Keep standard content layer
        'style_layer_weights': [0.3, 0.4, 0.3],  # Custom weights for kanagawa
        'use_raw_features': False
    }
}

# Kanagawa style definition
KANAGAWA_STYLE = {
    "dataset": "images/singles/wave-of-kanagawa.jpg",
    "single": True
}

# Local dataset variations for kanagawa experiments
KANAGAWA_DATASETS = {
    "impressionism_dry_run": {
        "content": {"dataset": "images/Impressionism", "fraction": 0.01},
        "style": KANAGAWA_STYLE
    },
    "impressionism_small": {
        "content": {"dataset": "images/Impressionism", "fraction": 0.07},
        "style": KANAGAWA_STYLE
    },
    "impressionism_medium": {
        "content": {"dataset": "images/Impressionism", "fraction": 0.10},
        "style": KANAGAWA_STYLE
    },
    "impressionism_large": {
        "content": {"dataset": "images/Impressionism", "fraction": 0.15},
        "style": KANAGAWA_STYLE
    },
    "voc_small": {
        "content": {"dataset": "images/VOC2012", "fraction": 0.05},
        "style": KANAGAWA_STYLE
    },
    "voc_medium": {
        "content": {"dataset": "images/VOC2012", "fraction": 0.08},
        "style": KANAGAWA_STYLE
    }
}

# Function to build kanagawa experiments with injected dependencies
def get_kanagawa_experiments(curricula):
    """Build kanagawa experiments with curricula dependency injected."""
    return {
    "kanagawa": {
        "model_size": "medium",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "kanagawa_dry_run": {
        "model_size": "small",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_dry_run"],
        "curriculum": {"stages": curricula["dry_run"]}
    },

    "high_kanagawa": {
        "model_size": "medium",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard_high_style"]}
    },

    "big_kanagawa": {
        "model_size": "big",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "small_kanagawa": {
        "model_size": "small",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "small_high_kanagawa": {
        "model_size": "small",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard_high_style"]}
    },

    "shallow_kanagawa": {
        "model_size": "medium",
        "layer_preset": "shallow",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "deep_kanagawa": {
        "model_size": "medium",
        "layer_preset": "deep",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "weighted_kanagawa": {
        "model_size": "medium",
        "layer_preset": "standard_weighted",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "deep_weighted_kanagawa": {
        "model_size": "medium",
        "layer_preset": "deep_weighted",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "mini_kanagawa": {
        "model_size": "small",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_large"],
        "curriculum": {"stages": curricula["standard_mega_style"]}
    },

    "super_kanagawa": {
        "model_size": "big",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["impressionism_large"],
        "curriculum": {"stages": curricula["standard_high_style"]}
    },

    # Additional variations using different datasets
    "kanagawa_voc": {
        "model_size": "medium",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["voc_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    "kanagawa_large_content": {
        "model_size": "big",
        "layer_preset": "standard_weighted",
        **KANAGAWA_DATASETS["impressionism_large"],
        "curriculum": {"stages": curricula["standard"]}
    },

    # Example using kanagawa-specific layer preset
    "kanagawa_custom_layers": {
        "model_size": "medium",
        "layer_preset": "kanagawa_optimized",  # Uses KANAGAWA_LAYER_PRESETS
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    }
    }
