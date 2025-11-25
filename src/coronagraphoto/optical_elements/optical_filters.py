"""Module holding color filters."""

from typing import final

import interpax
import jax.numpy as jnp
from jaxtyping import Array

from coronagraphoto.core.optical_elements import AbstractOpticalElement


@final
class OpticalFilter(AbstractOpticalElement):
    """A generic optical filter with wavelength-dependent throughput."""

    wavelengths_nm: Array
    transmittances: Array
    interp: interpax.Interpolator1D

    def __init__(self, wavelengths_nm: Array, transmittances: Array):
        """Initialize the optical filter."""
        self.wavelengths_nm = wavelengths_nm
        self.transmittances = transmittances
        self.interp = interpax.Interpolator1D(
            wavelengths_nm,
            transmittances,
            method="linear",
            extrap=jnp.array([0.0, 0.0]),
        )

    def get_throughput(self, wavelength_nm: float) -> Array:
        """Return the scalar throughput at a specific wavelength."""
        return self.interp(wavelength_nm)

    def apply(self, arr: Array, wavelength_nm: float):
        """Apply filter effect to the input array."""
        return self.interp(wavelength_nm) * arr
