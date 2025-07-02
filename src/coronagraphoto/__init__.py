"""Package for simulating coronagraphic observations."""

from .composite import CompositeObservation
from .detector import Detector
from .logger import logger
from .observation import Observation
from .observations import Observations
from .observing_scenario import ObservingScenario
from .post_processing import PostProcessing
from .processing_config import ProcessingConfig
from .settings import Settings

__all__ = [
    "logger",
    "Observation",
    "Observations",
    "ObservingScenario",
    "Settings",
    "PostProcessing",
    "ProcessingConfig",
    "CompositeObservation",
    "Detector",
]
