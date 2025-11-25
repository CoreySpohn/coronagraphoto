"""Defines an optical path through a telescope."""

import equinox as eqx

from coronagraphoto.core.optical_elements import AbstractOpticalElement
from coronagraphoto.optical_elements.coronagraph import Coronagraph


class OpticalPath(eqx.Module):
    """An optical path through a telescope.

    This class is used to create an optical path through a telescope. It is
    composed of a primary aperture, a sequence of attenuating elements
    (mirrors, filters, etc.), a coronagraph, and a detector.
    """

    primary: AbstractOpticalElement
    attenuating_elements: tuple[AbstractOpticalElement, ...]
    coronagraph: Coronagraph
    detector: AbstractOpticalElement

    def __init__(
        self,
        primary,
        attenuating_elements,
        coronagraph,
        detector,
    ):
        """Initialize the optical path."""
        self.primary = primary
        self.attenuating_elements = attenuating_elements
        self.coronagraph = coronagraph
        self.detector = detector

    def calculate_combined_attenuation(self, wavelength_nm: float) -> float:
        """Calculate the combined attenuation of the optical path at a specific wavelength."""
        combined_attenuation = 1.0
        # This loop gets unrolled when JIT compiled
        for element in self.attenuating_elements:
            combined_attenuation *= element.get_throughput(wavelength_nm)
        return combined_attenuation
