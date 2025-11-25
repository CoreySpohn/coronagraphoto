"""Module holding baisc throughput elements.

These represent things like mirrors that are in the optical path and may
attenuate the light, but do not provide significant image aberrations or
shape the PSF.
"""

from typing import final

import interpax
import jax.numpy as jnp
from jaxtyping import Array

from coronagraphoto.core.optical_elements import AbstractOpticalElement


@final
class ConstantThroughputElement(AbstractOpticalElement):
    """A generic constant throughput element with a constant throughput."""

    throughput: float

    def __init__(self, throughput: float):
        """Initialize the throughput element."""
        self.throughput = throughput

    def get_throughput(self, wavelength_nm: float) -> Array:
        """Return the scalar throughput at a specific wavelength."""
        return self.throughput

    def apply(self, arr: Array, wavelength: float):
        """Apply component effect to the input array."""
        return self.throughput * arr


@final
class LinearThroughputElement(AbstractOpticalElement):
    """A generic linear throughput element with wavelength-dependent throughput."""

    wavelengths_nm: Array
    throughputs: Array
    interp: interpax.Interpolator1D

    def __init__(self, wavelengths_nm: Array, throughputs: Array):
        """Initialize the throughput element."""
        self.wavelengths_nm = wavelengths_nm
        self.throughputs = throughputs
        self.interp = interpax.Interpolator1D(
            wavelengths_nm, throughputs, method="linear", extrap=jnp.array([0.0, 0.0])
        )

    def get_throughput(self, wavelength_nm: float) -> Array:
        """Return the scalar throughput at a specific wavelength."""
        return self.interp(wavelength_nm)

    def apply(self, arr: Array, wavelength: float):
        """Apply component effect to the input array."""
        return self.interp(wavelength) * arr
