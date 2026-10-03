# Training Curricula and Hyperparameters
# Shared training schedules and learning configurations

# HYPERPARAMS
low_rate  = 1e-4
med_rate  = 5e-4
high_rate = 1e-3
mini_rate  = 1e-5

# Style weights
# Rescaled on 2026-10-03 for the sum (Johnson et al.) reduction of the style
# loss: the old values (2e5, 8e4, 4e4, 2e4) divided by ~85,000. Untested
# starting points, as is tv_weight.
extra_high_style = 2.5
high_style = 1.0
med_style  = 0.5
low_style  = 0.25

tv_weight = 1e-6

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
        "tv_weight": tv_weight,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "standard": [{
        "epochs": 6,
        "lr": med_rate,
        "style_weight": med_style,
        "tv_weight": tv_weight,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "standard_high_style": [{
        "epochs": 6,
        "lr": med_rate,
        "style_weight": high_style,
        "tv_weight": tv_weight,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "long_hires_histyle": [{
        "res": 364,
        "epochs": 12,
        "lr": med_rate,
        "style_weight": high_style,
        "tv_weight": tv_weight,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "standard_mega_style": [{
        "epochs": 6,
        "lr": med_rate,
        "style_weight": extra_high_style,
        "tv_weight": tv_weight,
        "content_batch_size": 4,
        "style_batch_size": 1
    }],
    "long": [{
        "res": 256,
        "epochs": 4,
        "lr": med_rate,
        "style_weight": extra_high_style,
        "tv_weight": tv_weight,
        "content_batch_size": 4,
        "style_batch_size": 1
    }]
}
