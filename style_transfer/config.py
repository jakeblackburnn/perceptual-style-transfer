# STYLE TRANSFER TRAINING CONFIGURATION

    # HYPERPARAMS

# learning rates
low_rate  = 1e-4
med_rate  = 5e-4
high_rate = 1e-3

#style weights
extra_high_style = 2e5
high_style = 8e4
med_style  = 4e4
low_style  = 2e4


Models = {
    "dry_run": {
        "model_size": "small",
        "layer_preset": "standard",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.01,
        },

        "style": {
            "dataset": "images/Ukiyo_e",
            "fraction": 0.1,
        },

        "curriculum": {
            "stages": [
                {"res": 32, "epochs": 2, "lr": low_rate},
                {"res": 32, "epochs": 2, "lr": low_rate}
            ]
        }
    },

    "big_no_style_v1": {
        "model_size": "big",
        "layer_preset": "standard",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.02,
        },

        "style": {
            "dataset": "images/Ukiyo_e",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {
                    "epochs": 8, 
                    "lr": med_rate, 
                    "content_batch_size": 4, 
                    "style_batch_size": 1, 
                    "style_weight": 0
                },
            ]
        }
    },


    "medium_no_style_v1": {
        "model_size": "medium",
        "layer_preset": "standard",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.02,
        },

        "style": {
            "dataset": "images/Ukiyo_e",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {
                    "epochs": 6, 
                    "lr": med_rate, 
                    "content_batch_size": 4, 
                    "style_batch_size": 1, 
                    "style_weight": 0
                },
            ]
        }
    },

    "medium_no_style_deep_loss_v1": {
        "model_size": "medium",
        "layer_preset": "deep",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.02,
        },

        "style": {
            "dataset": "images/Ukiyo_e",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {
                    "epochs": 6, 
                    "lr": med_rate, 
                    "content_batch_size": 4, 
                    "style_batch_size": 1, 
                    "style_weight": 0
                },
            ]
        }
    },

    "small_no_style_v1": {
        "model_size": "small",
        "layer_preset": "standard",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.02,
        },

        "style": {
            "dataset": "images/Ukiyo_e",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {
                    "epochs": 4, 
                    "lr": med_rate, 
                    "content_batch_size": 4, 
                    "style_batch_size": 1, 
                    "style_weight": 0
                },
            ]
        }
    },

    "basic_small_impressionism_v1": {
        "model_size": "small",
        "layer_preset": "standard",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.01,
        },

        "style": {
            "dataset": "images/Impressionism",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {"epochs": 4, "lr": low_rate},
            ]
        }
    },

    "basic_small_impressionism_deep_v1": {
        "model_size": "small",
        "layer_preset": "deep",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.01,
        },

        "style": {
            "dataset": "images/Impressionism",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {"epochs": 4, "lr": low_rate, "style_weight": low_style },
            ]
        }
    },

    "medium_impressionism_weighted_v1": {
        "model_size": "medium",
        "layer_preset": "standard_weighted",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.01,
        },

        "style": {
            "dataset": "images/Impressionism",
            "fraction": 0.01,
        },

        "curriculum": {
            "stages": [
                {"epochs": 4, "lr": low_rate, "style_weight": low_style},
            ]
        }
    },

    "medium_ukiyo_weighted_v1": {
        "model_size": "medium",
        "layer_preset": "standard_weighted",

        "content": {
            "dataset": "images/VOC2012",
            "fraction": 0.01,
        },

        "style": {
            "dataset": "images/Ukiyo_e",
            "fraction": 0.2,
        },

        "curriculum": {
            "stages": [
                {"epochs": 4, "lr": low_rate, "style_weight": low_style},
            ]
        }
    }
}
