"""Per-pixel photon budgets, detector observations and aperture exposure times."""

from coronagraphoto.calculator import (
    ExposureTime,
    PostProcessing,
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
    "ExposureTime",
    "PostProcessing",
    "build_exposure",
    "expected",
    "exposure_time",
    "observe",
    "photon_rates",
    "snr",
]
