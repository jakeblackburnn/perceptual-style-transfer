# Simple explicit imports for style experiments
from .kanagawa import get_kanagawa_experiments
from .port_of_collioure import get_collioure_experiments
from .starry_night import get_starry_night_experiments
from .colors1 import get_colors1_experiments
from .colors2 import get_colors2_experiments
from ..curricula import CURRICULA

# Build experiments with dependency injection - simple and explicit
ALL_EXPERIMENTS = {}
ALL_EXPERIMENTS.update(get_kanagawa_experiments(CURRICULA))
ALL_EXPERIMENTS.update(get_collioure_experiments(CURRICULA))
ALL_EXPERIMENTS.update(get_starry_night_experiments(CURRICULA))
ALL_EXPERIMENTS.update(get_colors1_experiments(CURRICULA))
ALL_EXPERIMENTS.update(get_colors2_experiments(CURRICULA))
