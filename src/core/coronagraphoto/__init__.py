"""
Coronagraphoto v2: Observatory & Reduction Framework

A medium-fidelity coronagraph observation simulation tool built around
a functional approach for light path simulation.
"""

__version__ = "2.0.0-dev"

from .observation import Target, Observation, ObservationSequence
from .data_models import IntermediateData, PropagationContext
from .observatory import Observatory
from .reduction import ReductionPipeline, ReductionStep

__all__ = [
    "Target",
    "Observation", 
    "ObservationSequence",
    "IntermediateData",
    "PropagationContext",
    "Observatory",
    "ReductionPipeline",
    "ReductionStep",
]