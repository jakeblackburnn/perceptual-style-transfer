# STYLE TRANSFER TRAINING CONFIGURATION
#
# defines optimizer and training curriculum hyperparams



    # OPTIMIZER

import torch
optimizer = torch.optim.Adam



    # HYPERPARAMS

# learning rates
low_rate  = 1e-4
med_rate  = 5e-4
high_rate = 1e-3

#style weights
high_style = 1e6
# default is 1e5 (medium)
low_style = 5e4



    # CURRICULA

Curricula = {

        # dry run for testing basic functionality
    "dry_run": {
        "stages": [
            {"resolution": 64, "batch_size": 4, "epochs": 1, "lr": med_rate}
        ]
    },

    "no_style": {
        "stages": [
            {"resolution": 256, "batch_size": 1, "epochs": 8, "lr": med_rate, "style_weight": 0},
            {"resolution": 512, "batch_size": 1, "epochs": 4, "lr": low_rate, "style_weight": 0}
        ]
    },



            # BASIC CURRICULUM - small scale straitforward training
    

    # simple small training curriculum
    "basic": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 4, "lr": med_rate}
        ]
    },
    # simple small training curriculum w/ high style weight
    "basic_high_style": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 4, "lr": med_rate, "style_weight": high_style}
        ]
    },
    # simple small training curriculum w/ low style weight
    "basic_low_style": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 4, "lr": med_rate, "style_weight": low_style}
        ]
    },
    # simple small training curriculum w/ bigger batches
    "basic_big_batch": {
        "stages": [
            {"resolution": 224, "batch_size": 8, "epochs": 4, "lr": med_rate}
        ]
    },
    # simple small training curriculum w/ smaller batches
    "basic_small_batch": {
        "stages": [
            {"resolution": 224, "batch_size": 2, "epochs": 4, "lr": med_rate}
        ]
    },
    # simple small training curriculum w/ higher LR
    "basic_high_rate": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 4, "lr": high_rate}
        ]
    },
    # simple small training curriculum w/ lower LR
    "basic_low_rate": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 4, "lr": low_rate}
        ]
    },



            # STANDARD CURRICULUM - larger scale straitforward training

    # standard training
    "standard": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 12, "lr": low_rate},
        ]
    },

    # high style weight
    "standard_high_style": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 12, "lr": low_rate, "style_weight": high_style},
        ]
    },

    # high style weight
    "standard_low_style": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 12, "lr": low_rate, "style_weight": low_style},
        ]
    },

    # high resolution 
    "standard_high_res": {
        "stages": [
            {"resolution": 512, "batch_size": 4, "epochs": 12, "lr": low_rate},
        ]
    },

    # low resolution
    "standard_low_res": {
        "stages": [
            {"resolution": 128, "batch_size": 4, "epochs": 12, "lr": low_rate},
        ]
    },



        # ADVANCED CURRICULUM - mixed training stages testing various curriculum strategies

    # increasing difficulty (resolution)

    "easy_to_hard": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 6, "lr": high_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 4, "lr": med_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 2, "lr": low_rate},
        ]
    },

    # decreasing difficulty (exact reverse of easy to hard stages)
    "hard_to_easy": {
        "stages": [
            {"resolution": 512, "batch_size": 4, "epochs": 2, "lr": low_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 4, "lr": med_rate},
            {"resolution": 256, "batch_size": 4, "epochs": 6, "lr": high_rate},
        ]
    },

    # cyclical difficulty
    "cyclic": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 2, "lr": high_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 1, "lr": med_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 1, "lr": low_rate},

            {"resolution": 256, "batch_size": 4, "epochs": 2, "lr": med_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 1, "lr": med_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 1, "lr": low_rate},

            {"resolution": 256, "batch_size": 4, "epochs": 2, "lr": med_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 1, "lr": low_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 1, "lr": low_rate}
        ]
    },

    # increasing style weight
    "increasing_style": {
        "stages": [
            {"resolution": 256, "batch_size": 1, "epochs": 4, "lr": med_rate, "style_weight": 0},
            {"resolution": 256, "batch_size": 4, "epochs": 4, "lr": med_rate},
            {"resolution": 256, "batch_size": 4, "epochs": 4, "lr": med_rate, "style_weight": high_style}
        ]
    }
}
