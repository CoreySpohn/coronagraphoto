"""Re-exports of hardware abstractions from :mod:`optixstuff`.

Detector, primary, throughput, and filter classes live in optixstuff so
both coronagraphoto (image simulation) and jaxedith (ETC) share one set
of hardware models. This module is a thin compatibility shim -- prefer
importing directly from ``optixstuff`` in new code.
"""

from optixstuff import (
    AbstractDetector,
    AbstractOpticalElement,
    AbstractPrimary,
    ConstantThroughputElement,
    Detector,
    LinearThroughputElement,
    OpticalFilter,
    SimpleDetector,
    SimplePrimary,
)
from optixstuff.detector import (
    simulate_cic,
    simulate_dark_current,
    simulate_read_noise,
)

__all__ = [
    "AbstractDetector",
    "AbstractOpticalElement",
    "AbstractPrimary",
    "ConstantThroughputElement",
    "Detector",
    "LinearThroughputElement",
    "OpticalFilter",
    "SimpleDetector",
    "SimplePrimary",
    "simulate_cic",
    "simulate_dark_current",
    "simulate_read_noise",
]
