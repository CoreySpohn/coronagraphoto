"""Optical elements for coronagraphoto."""

from coronagraphoto.optical_elements.coronagraph import Coronagraph, from_yippy
from coronagraphoto.optical_elements.detector import (
    AbstractDetector,
    Detector,
    SimpleDetector,
)
from coronagraphoto.optical_elements.optical_filters import OpticalFilter
from coronagraphoto.optical_elements.primary import PrimaryAperture
from coronagraphoto.optical_elements.throughput_elements import (
    ConstantThroughputElement,
    LinearThroughputElement,
)

__all__ = [
    "Coronagraph",
    "from_yippy",
    "AbstractDetector",
    "SimpleDetector",
    "Detector",
    "OpticalFilter",
    "PrimaryAperture",
    "ConstantThroughputElement",
    "LinearThroughputElement",
]
