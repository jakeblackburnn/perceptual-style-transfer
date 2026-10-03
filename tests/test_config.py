import numbers

import pytest

from style_transfer.config import Models
from style_transfer.config.layer_presets import LAYER_PRESETS, get_layer_preset


@pytest.mark.parametrize("name", sorted(Models))
def test_experiment_is_well_formed(name):
    config = Models[name]

    # raises KeyError if the experiment names a preset that doesn't exist
    get_layer_preset(config["layer_preset"])

    stages = config["curriculum"]["stages"]
    assert len(stages) > 0
    for stage in stages:
        assert isinstance(stage["lr"], numbers.Real)
        assert isinstance(stage["epochs"], numbers.Real)


@pytest.mark.parametrize("name", sorted(LAYER_PRESETS))
def test_layer_weights_match_style_layers(name):
    preset = LAYER_PRESETS[name]
    assert len(preset["style_layer_weights"]) == len(preset["style_layers"])


def test_unknown_layer_preset_raises():
    with pytest.raises(KeyError):
        get_layer_preset("no_such_preset")
