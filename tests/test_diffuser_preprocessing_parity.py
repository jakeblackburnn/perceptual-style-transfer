"""Regression guard for the [0, 1] <-> [-1, 1] boundary diffuser introduces.

style_transfer/ keeps all image tensors in [0, 1] range throughout (see
tests/test_preprocessing_parity.py). diffuser/schedule.py's
to_diffusion_space/from_diffusion_space are the single place that boundary
is crossed to satisfy diffusers schedulers' [-1, 1] convention; this guards
that conversion against the same class of silent scaling bug CLAUDE.md's
Goals section documents (*255 train/inference mismatch).
"""

import torch

from diffuser.schedule import to_diffusion_space, from_diffusion_space


def test_round_trip_identity():
    x01 = torch.rand(2, 3, 8, 8)
    assert torch.allclose(from_diffusion_space(to_diffusion_space(x01)), x01, atol=1e-6)


def test_to_diffusion_space_known_values():
    x01 = torch.tensor([0.0, 0.5, 1.0])
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(to_diffusion_space(x01), expected)


def test_from_diffusion_space_clamps_out_of_range_input():
    x_pm1 = torch.tensor([-1.5, 0.0, 1.5])
    result = from_diffusion_space(x_pm1)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
