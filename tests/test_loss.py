import pytest
import torch
import torch.nn as nn

import style_transfer.feature_extractors.vgg as vgg_module
from style_transfer.loss import gram_matrix, perceptual_loss, total_variation


def test_gram_matrix_known_value():
    # feature_map: batch=1, channels=2, h=1, w=2
    # channel 0 = [1, 2], channel 1 = [3, 4]
    feature_map = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    assert feature_map.shape == (1, 2, 1, 2)

    gram = gram_matrix(feature_map)

    # features = [[1, 2], [3, 4]]; features @ features.T = [[5, 11], [11, 25]]
    # normalized by c*h*w = 2*1*2 = 4
    expected = torch.tensor([[[1.25, 2.75], [2.75, 6.25]]])
    assert torch.allclose(gram, expected)


def test_gram_matrix_is_symmetric():
    feature_map = torch.rand(2, 4, 8, 8)
    gram = gram_matrix(feature_map)
    assert torch.allclose(gram, gram.transpose(1, 2), atol=1e-6)


# --- perceptual_loss, scored through a fake feature extractor ---------------
# No pretrained weights: the stub below is installed as the process-global
# VGG that perceptual_loss reads, and its "features" are simple functions of
# the input so every expected value can be computed by hand.

class FakeFeatureExtractor(nn.Module):
    def __init__(self, style_layer_weights=None, use_raw_features=False):
        super().__init__()
        self.preset_config = {
            'style_layers': ['s1', 's2'],
            'content_layer': 'c',
            'use_raw_features': use_raw_features,
        }
        if style_layer_weights is not None:
            self.preset_config['style_layer_weights'] = style_layer_weights

    def forward(self, x):
        return {'s1': x, 's2': 2 * x, 'c': x}


@pytest.fixture
def fake_vgg(monkeypatch):
    # monkeypatch restores the previous global after each test
    def install(**kwargs):
        monkeypatch.setattr(vgg_module, '_vgg_model', FakeFeatureExtractor(**kwargs))
    return install


# generated: batch=1, channels=2, h=1, w=2; channel 0 = [1, 2], channel 1 = [3, 4]
# its gram is [[1.25, 2.75], [2.75, 6.25]] (see test_gram_matrix_known_value)
GENERATED = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
ZEROS = torch.zeros(1, 2, 1, 2)

# against all-zero targets:
#   content = mean(1, 4, 9, 16)                       = 7.5
#   s1      = 1.25^2 + 2 * 2.75^2 + 6.25^2 (the SUM)  = 55.75
#   s2      = features doubled -> gram x4 -> x16      = 892.0
CONTENT, S1, S2 = 7.5, 55.75, 892.0


def test_perceptual_loss_hand_computed_value(fake_vgg):
    fake_vgg()

    loss = perceptual_loss(GENERATED, ZEROS, ZEROS, content_weight=1.0, style_weight=1.0)
    assert loss.item() == pytest.approx(CONTENT + S1 + S2)

    loss = perceptual_loss(GENERATED, ZEROS, ZEROS, content_weight=2.0, style_weight=0.5)
    assert loss.item() == pytest.approx(2.0 * CONTENT + 0.5 * (S1 + S2))


def test_style_term_is_averaged_over_generated_style_pairs(fake_vgg):
    fake_vgg()

    # two style images: zeros (full style loss) and the generated image itself (none)
    styles = torch.cat([ZEROS, GENERATED])
    loss = perceptual_loss(GENERATED, ZEROS, styles, content_weight=1.0, style_weight=1.0)
    assert loss.item() == pytest.approx(CONTENT + (S1 + S2) / 2)


def test_style_layer_weights_scale_each_layer(fake_vgg):
    fake_vgg(style_layer_weights=[1.0, 0.5])
    loss = perceptual_loss(GENERATED, ZEROS, ZEROS, content_weight=1.0, style_weight=1.0)
    assert loss.item() == pytest.approx(CONTENT + S1 + 0.5 * S2)

    # a zero weight removes that layer's contribution
    fake_vgg(style_layer_weights=[1.0, 0.0])
    loss = perceptual_loss(GENERATED, ZEROS, ZEROS, content_weight=1.0, style_weight=1.0)
    assert loss.item() == pytest.approx(CONTENT + S1)


def test_style_layer_weights_apply_to_raw_features(fake_vgg):
    fake_vgg(style_layer_weights=[1.0, 0.5], use_raw_features=True)

    # raw features keep the mean reduction: s1 = mean(1, 4, 9, 16) = 7.5, s2 = 4 * s1 = 30
    loss = perceptual_loss(GENERATED, ZEROS, ZEROS, content_weight=0.0, style_weight=1.0)
    assert loss.item() == pytest.approx(7.5 + 0.5 * 30.0)


def test_tv_weight_adds_hand_computed_total_variation(fake_vgg):
    fake_vgg()

    # image 0 = [[1, 2], [4, 8]] in both channels, image 1 = zeros
    #   vertical:   (4-1)^2 + (8-2)^2 = 45
    #   horizontal: (2-1)^2 + (8-4)^2 = 17
    # two channels -> 124 for image 0, 0 for image 1, batch average = 62
    image = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    generated = torch.stack([torch.stack([image, image]), torch.zeros(2, 2, 2)])
    targets = torch.zeros(2, 2, 2, 2)
    assert total_variation(generated).item() == pytest.approx(62.0)

    # a small style_weight keeps the other terms small enough that float32
    # doesn't blur the difference
    without_tv = perceptual_loss(generated, targets, targets, style_weight=0.01)
    with_tv = perceptual_loss(generated, targets, targets, style_weight=0.01, tv_weight=0.1)
    assert (with_tv - without_tv).item() == pytest.approx(6.2, rel=1e-4)


def test_gradients_flow_to_generated_image(fake_vgg):
    fake_vgg()

    generated = GENERATED.clone().requires_grad_(True)
    loss = perceptual_loss(generated, ZEROS, ZEROS, tv_weight=1e-6)
    loss.backward()

    assert generated.grad is not None
    assert torch.all(generated.grad != 0)
