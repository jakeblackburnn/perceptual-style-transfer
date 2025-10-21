starry_night = {
    "dataset": "artifacts/images/singles/starry-night.jpg",
    "single": True
}

dataset = {
    "content": {"dataset": "artifacts/images/Pointillism", "fraction": 1.0},
    "style": starry_night,
}

def get_starry_night_experiments(curricula):
    return {
        "starry_night": {
            "model_size": "medium",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["standard"]},
        },
        "high_starry_night": {
            "model_size": "medium",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["standard_mega_style"]},
        },
        "mini_starry_night": {
            "model_size": "small",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["long_hires_histyle"]},
        },
    }
