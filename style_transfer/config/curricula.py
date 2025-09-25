# Training Curricula and Hyperparameters
# Shared training schedules and learning configurations

# HYPERPARAMS
low_rate  = 1e-4
med_rate  = 5e-4
high_rate = 1e-3
mini_rate  = 1e-5

# Style weights
extra_high_style = 2e5
high_style = 8e4
med_style  = 4e4
low_style  = 2e4
mini_style  = 10
tiny_style  = 0.25

# CURRICULUM PRESETS
CURRICULA = {
    "dry_run": [
        {"res": 32, "epochs": 2, "lr": low_rate},
        {"res": 32, "epochs": 2, "lr": low_rate}
    ],
    "quick": [{
        "epochs": 4,
        "lr": med_rate,
        "style_weight": med_style,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "standard": [{
        "epochs": 6,
        "lr": med_rate,
        "style_weight": med_style,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "standard_high_style": [{
        "epochs": 6,
        "lr": med_rate,
        "style_weight": high_style,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "standard_mega_style": [{
        "epochs": 6,
        "lr": med_rate,
        "style_weight": extra_high_style,
        "content_batch_size": 4,
        "style_batch_size": 1
    }]
}
