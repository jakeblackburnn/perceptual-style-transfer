import torch

from style_transfer.models import StyleTransferModel


def test_forward_pass_shapes():
    x = torch.rand(1, 3, 64, 64)
    for size in ["small", "medium", "big"]:
        model = StyleTransferModel(size_config=size).eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, f"size={size}: expected {x.shape}, got {out.shape}"


def test_output_is_in_unit_range():
    # models.py's final activation rescales tanh's [-1, 1] output to [0, 1];
    # inference/training preprocessing both feed [0, 1]-range input to match.
    x = torch.rand(1, 3, 64, 64)
    model = StyleTransferModel(size_config="small").eval()
    with torch.no_grad():
        out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_unknown_size_config_raises():
    try:
        StyleTransferModel(size_config="huge")
        assert False, "expected ValueError for unknown size_config"
    except ValueError:
        pass
