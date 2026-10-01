"""Config dataclasses for diffuser/ experiments.

Unlike style_transfer/config/curricula.py's stage-list shape (built for
progressive resolution / style-weight ramps across a feed-forward training
run), diffusion training here has no multi-phase structure: one fixed
image_size and one set of hyperparameters for the whole run. A flat,
type-checked dataclass is simpler and self-documenting for that shape.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UNetConfig:
    layers_per_block: int = 2
    block_out_channels: tuple = (64, 128, 192, 256)
    down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
    up_block_types: tuple = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")


@dataclass(frozen=True)
class SamplingConfig:
    num_inference_steps: int = 50


@dataclass(frozen=True)
class DiffusionExperimentConfig:
    image_size: int = 128
    unet: UNetConfig = field(default_factory=UNetConfig)

    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    ema_decay: float = 0.9995
    grad_clip_norm: float = 1.0

    # perceptual_loss is only scored against x0_pred for timesteps below
    # perceptual_cutoff_frac * num_train_timesteps -- at higher-noise
    # timesteps x0_pred is too poor an estimate for the perceptual gradient
    # to be a useful (rather than actively harmful) training signal.
    perceptual_cutoff_frac: float = 0.2
    perceptual_content_weight: float = 1.0
    perceptual_style_weight: float = 1e5
    perceptual_loss_weight: float = 1e-2
    layer_preset: str = "standard"

    lr: float = 2e-4
    epochs: int = 10
    content_batch_size: int = 4

    content: dict = field(default_factory=dict)  # {"dataset": path, "fraction": frac}
    style: dict = field(default_factory=dict)  # {"dataset": path, "single": True}

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
