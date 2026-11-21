"""Per-pixel photon budgets, detector observations and aperture rates."""

from coronagraphoto.aperture import (
    ApertureRecord,
    ApertureResponse,
    aperture_rates,
    aperture_record,
    aperture_response,
    scene_aperture_rates,
)
from coronagraphoto.simulation import (
    build_exposure,
    expected,
    observe,
    photon_rates,
)

__all__ = [
    "ApertureRecord",
    "ApertureResponse",
    "aperture_rates",
    "aperture_record",
    "aperture_response",
    "build_exposure",
    "expected",
    "observe",
    "photon_rates",
    "scene_aperture_rates",
]
