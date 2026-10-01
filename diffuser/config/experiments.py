"""Named diffusion experiment configs.

Mirrors style_transfer/config/styles/*.py's named-experiment-dict pattern,
just with a flat DiffusionExperimentConfig per name instead of a curriculum
dict (see schema.py for why: diffusion training here has no multi-phase
structure to stage).
"""

from .schema import DiffusionExperimentConfig
from .unet_presets import get_unet_preset

FLOWERS_KANAGAWA_DRY_RUN = DiffusionExperimentConfig(
    image_size=32,
    unet=get_unet_preset("small"),
    epochs=1,
    content_batch_size=2,
    content={"dataset": "artifacts/images/content/flowers", "fraction": 0.1},
    style={"dataset": "artifacts/images/singles/wave-of-kanagawa.jpg", "single": True},
)

FLOWERS_KANAGAWA = DiffusionExperimentConfig(
    image_size=128,
    unet=get_unet_preset("standard"),
    epochs=20,
    content_batch_size=4,
    content={"dataset": "artifacts/images/content/flowers", "fraction": 1.0},
    style={"dataset": "artifacts/images/singles/wave-of-kanagawa.jpg", "single": True},
)

ALL_DIFFUSION_EXPERIMENTS = {
    "flowers_kanagawa_dry_run": FLOWERS_KANAGAWA_DRY_RUN,
    "flowers_kanagawa": FLOWERS_KANAGAWA,
}
