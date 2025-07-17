import torch

optimizer = torch.optim.Adam

low_rate  = 1e-4
med_rate  = 5e-4
high_rate = 1e-3

Curricula = {

        # dry run for testing basic functionality
    "dry_run": {
        "stages": [
            {"resolution": 64, "batch_size": 4, "epochs": 1, "lr": med_rate}
        ]
    },



            # BASIC CURRICULUM - small scale straitforward training
    

    # simple small training curriculum
    "basic": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 8, "lr": med_rate}
        ]
    },
    # simple small training curriculum w/ bigger batches
    "basic_big_batch": {
        "stages": [
            {"resolution": 224, "batch_size": 8, "epochs": 8, "lr": med_rate}
        ]
    },
    # simple small training curriculum w/ smaller batches
    "basic_small_batch": {
        "stages": [
            {"resolution": 224, "batch_size": 2, "epochs": 8, "lr": med_rate}
        ]
    },
    # simple small training curriculum w/ higher LR
    "basic_high_rate": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 8, "lr": high_rate}
        ]
    },
    # simple small training curriculum w/ lower LR
    "basic_low_rate": {
        "stages": [
            {"resolution": 224, "batch_size": 4, "epochs": 8, "lr": low_rate}
        ]
    },



            # STANDARD CURRICULUM - medium scale straitforward training

    # standard training
    "standard": {
        "stages": [
            {"resolution": 512, "batch_size": 4, "epochs": 16, "lr": low_rate},
        ]
    },

    # high resolution 
    "standard_high_res": {
        "stages": [
            {"resolution": 768, "batch_size": 4, "epochs": 16, "lr": low_rate},
        ]
    },

    # low resolution
    "standard_low_res": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 16, "lr": low_rate},
        ]
    },



        # ADVANCED CURRICULUM - large scale mixed training stages w/ varying complexity

    # increasing difficulty
    "easy_to_hard": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 8, "lr": high_rate},
            {"resolution": 256, "batch_size": 4, "epochs": 8, "lr": med_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 8, "lr": med_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 8, "lr": low_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 4, "lr": low_rate},
            {"resolution": 768, "batch_size": 4, "epochs": 4, "lr": low_rate},
        ]
    },
    # decreasing difficulty (exact inverse of easy to hard)
    "hard_to_easy": {
        "stages": [
            {"resolution": 768, "batch_size": 4, "epochs": 4, "lr": low_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 4, "lr": low_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 8, "lr": low_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 8, "lr": med_rate},
            {"resolution": 256, "batch_size": 4, "epochs": 8, "lr": med_rate},
            {"resolution": 256, "batch_size": 4, "epochs": 8, "lr": high_rate},
        ]
    },
    # cyclical difficulty
    "cyclic": {
        "stages": [
            {"resolution": 256, "batch_size": 4, "epochs": 4, "lr": high_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 4, "lr": med_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 4, "lr": low_rate},

            {"resolution": 256, "batch_size": 4, "epochs": 4, "lr": med_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 4, "lr": low_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 4, "lr": low_rate},

            {"resolution": 256, "batch_size": 4, "epochs": 4, "lr": low_rate},
            {"resolution": 384, "batch_size": 4, "epochs": 4, "lr": low_rate},
            {"resolution": 512, "batch_size": 4, "epochs": 4, "lr": low_rate}
        ]
    }
}

