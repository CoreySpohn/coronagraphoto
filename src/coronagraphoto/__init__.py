"""Per-pixel photon budgets, detector observations and aperture exposure times."""

from coronagraphoto.aperture import (
    ApertureCalibration,
    ApertureRecord,
    aperture_calibration,
    aperture_rates,
    aperture_record,
)
from coronagraphoto.calculator import (
    ExposureTime,
    ResidualFractions,
    exposure_time,
    snr,
)
from coronagraphoto.simulation import (
    build_exposure,
    expected,
    observe,
    photon_rates,
)

__all__ = [
    "ApertureCalibration",
    "ApertureRecord",
    "ExposureTime",
    "ResidualFractions",
    "aperture_calibration",
    "aperture_rates",
    "aperture_record",
    "build_exposure",
    "expected",
    "exposure_time",
    "observe",
    "photon_rates",
    "snr",
]
