"""Multi-device scientific rendering verification package (stdlib only)."""
from .service import (
    render_job,
    available_scenes,
    reproducibility_config,
    discover_devices,
    profile_catalog,
    PRECISION_CONFIGS,
    MAX_RES,
    DEFAULT_RES,
)

__all__ = [
    "render_job",
    "available_scenes",
    "reproducibility_config",
    "discover_devices",
    "profile_catalog",
    "PRECISION_CONFIGS",
    "MAX_RES",
    "DEFAULT_RES",
]
