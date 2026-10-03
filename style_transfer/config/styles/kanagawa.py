# Kanagawa Style Configuration
# All kanagawa-related experiments, datasets, and configurations

# Dependencies will be injected by auto-discovery system

# Kanagawa style definition
KANAGAWA_STYLE = {
    "dataset": "artifacts/images/singles/wave-of-kanagawa.jpg",
    "single": True
}

# Local dataset variations for kanagawa experiments
KANAGAWA_DATASETS = {
    "impressionism_dry_run": {
        "content": {"dataset": "artifacts/images/Impressionism", "fraction": 0.01},
        "style": KANAGAWA_STYLE
    },
    "impressionism_small": {
        "content": {"dataset": "artifacts/images/Impressionism", "fraction": 0.07},
        "style": KANAGAWA_STYLE
    },
    "impressionism_medium": {
        "content": {"dataset": "artifacts/images/Impressionism", "fraction": 0.10},
        "style": KANAGAWA_STYLE
    },
    "impressionism_large": {
        "content": {"dataset": "artifacts/images/Impressionism", "fraction": 0.15},
        "style": KANAGAWA_STYLE
    },
    "voc_small": {
        "content": {"dataset": "artifacts/images/VOC2012", "fraction": 0.05},
        "style": KANAGAWA_STYLE
    },
    "voc_medium": {
        "content": {"dataset": "artifacts/images/VOC2012", "fraction": 0.08},
        "style": KANAGAWA_STYLE
    },
    "voc_full": {
        "content": {"dataset": "artifacts/images/VOC2012", "fraction": 1.0},  # ~23,000 images (jpg + png)
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

    # Example using the kanagawa-tuned layer preset
    "kanagawa_custom_layers": {
        "model_size": "medium",
        "layer_preset": "kanagawa_optimized",
        **KANAGAWA_DATASETS["impressionism_small"],
        "curriculum": {"stages": curricula["standard"]}
    },

    # Long run on all of VOC2012: ~23,000 steps at batch 4 over 4 epochs
    "kanagawa_long": {
        "model_size": "small",
        "layer_preset": "standard",
        **KANAGAWA_DATASETS["voc_full"],
        "curriculum": {"stages": curricula["long"]}
    }
    }
