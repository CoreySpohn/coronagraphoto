"""Per-pixel photon budgets, detector observations and aperture rates."""

from coronagraphoto.aperture import (
    AperturePosition,
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
    planet_source_support,
)

__all__ = [
    "AperturePosition",
    "ApertureRecord",
    "ApertureResponse",
    "aperture_rates",
    "aperture_record",
    "aperture_response",
    "build_exposure",
    "expected",
    "observe",
    "photon_rates",
    "planet_source_support",
    "scene_aperture_rates",
]
