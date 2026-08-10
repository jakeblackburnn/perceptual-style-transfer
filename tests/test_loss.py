import torch

from style_transfer.loss import gram_matrix


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
