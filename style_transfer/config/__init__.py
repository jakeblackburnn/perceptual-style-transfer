# Config module initialization
# Auto-discovery and main experiment interface

from .styles import ALL_EXPERIMENTS

# Expose experiments as Models for backward compatibility
Models = ALL_EXPERIMENTS

# Also expose individual modules for direct access if needed
from . import curricula
from . import layer_presets