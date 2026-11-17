"""Multi-device scientific rendering verification package (stdlib only)."""
from .service import (
    render_job,
    available_scenes,
    reproducibility_config,
    discover_devices,
    MAX_RES,
    DEFAULT_RES,
)

__all__ = [
    "render_job",
    "available_scenes",
    "reproducibility_config",
    "discover_devices",
    "MAX_RES",
    "DEFAULT_RES",
]
