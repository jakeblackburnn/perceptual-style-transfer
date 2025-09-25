port_of_colliore = {
    "dataset": "images/singles/port-of-collioure.jpg",
    "single": True
}

dataset = {
    "content": {"dataset": "images/Impressionism", "fraction": 0.07},
    "style": port_of_colliore,
}

def get_collioure_experiments(curricula):
    return {
        "port_of_collioure": {
            "model_size": "medium",
            "layer_preset": "standard",
            **dataset,
            "curriculum": { "stages": curricula["standard_high_style"]},
        },
        "shallow_collioure": {
            "model_size": "small",
            "layer_preset": "shallow",
            **dataset,
            "curriculum": { "stages": curricula["standard_high_style"]},
        },
    }

