colors2= {
    "dataset": "images/singles/colors2.jpg",
    "single": True
}

dataset = {
    "content": {"dataset": "images/Impressionism", "fraction": 0.07},
    "style": colors2,
}

def get_colors2_experiments(curricula):
    return {
        "colors2": {
            "model_size": "medium",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["standard_high_style"]},
        },
        "mini_colors2": {
            "model_size": "small",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["long_hires_histyle"]},
        },
    }
