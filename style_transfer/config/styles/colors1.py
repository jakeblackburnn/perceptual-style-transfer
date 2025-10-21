colors1 = {
    "dataset": "artifacts/images/singles/colors1.jpg",
    "single": True
}

dataset = {
    "content": {"dataset": "artifacts/images/Impressionism", "fraction": 0.07},
    "style": colors1,
}

big_dataset = {
    "content": {"dataset": "artifacts/images/Impressionism", "fraction": 0.2},
    "style": colors1,
}

def get_colors1_experiments(curricula):
    return {
        "colors1": {
            "model_size": "medium",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["standard"]},
        },
        "super_colors1": {
            "model_size": "big",
            "layer_preset": "standard",
            **big_dataset,
            "curriculum": { "stages": curricula["standard_high_style"]},
        },
        "mini_colors1": {
            "model_size": "small",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["long_hires_histyle"]},
        },
    }
