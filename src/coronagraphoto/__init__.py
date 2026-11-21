"""Per-pixel photon budgets, detector observations and aperture exposure times."""

from coronagraphoto.aperture import (
    ApertureRecord,
    ApertureResponse,
    aperture_rates,
    aperture_record,
    aperture_response,
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
]
