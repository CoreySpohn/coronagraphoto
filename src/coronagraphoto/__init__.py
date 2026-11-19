"""Per-pixel photon budgets, detector observations and aperture exposure times."""

from coronagraphoto.aperture import (
    ApertureRecord,
    ApertureResponse,
    aperture_rates,
    aperture_record,
    aperture_response,
)
from coronagraphoto.calculator import (
    ExposureTime,
    ResidualFractions,
    exposure_time,
    snr,
)
from coronagraphoto.simulation import (
    BuiltExposure,
    build_exposure,
    expected,
    observe,
    photon_rates,
)

__all__ = [
    "ApertureRecord",
    "ApertureResponse",
    "BuiltExposure",
    "ExposureTime",
    "ResidualFractions",
    "aperture_rates",
    "aperture_record",
    "aperture_response",
    "build_exposure",
    "expected",
    "exposure_time",
    "observe",
    "photon_rates",
    "snr",
]
