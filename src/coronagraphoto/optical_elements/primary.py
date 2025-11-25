"""Module holding the primary aperture."""

from typing import final

import jax.numpy as jnp
from jaxtyping import Array

from coronagraphoto.core.optical_elements import AbstractOpticalElement


@final
class PrimaryAperture(AbstractOpticalElement):
    """Generic primary aperture of the telescope."""

    diameter_m: float
    area_m2: float
    obscuration_factor: float

    def __init__(self, diameter_m: float, obscuration_factor: float = 0.0):
        """Initialize the primary aperture."""
        self.diameter_m = diameter_m
        self.area_m2 = jnp.pi * (diameter_m / 2) ** 2 * (1 - obscuration_factor)
        self.obscuration_factor = obscuration_factor

    def apply(self, inc_flux: float | Array, wavelength: float):
        """Multiply the incident flux by the area of the primary aperture.

        Note that currently there's no wavelength dependence, but that may
        change in the future.

        Args:
            inc_flux: The incident flux in ph/s/m^2/nm.
            wavelength: The wavelength in nm (placeholder for consistency).

        Returns:
            The outgoing flux in ph/s/nm.
        """
        return self.area_m2 * inc_flux
