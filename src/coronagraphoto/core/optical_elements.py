"""Module holding all standard optical elements."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp


class AbstractOpticalElement(eqx.Module):
    """The base class for all optical elements.

    This class is used to create all optical elements.
    """

    @abstractmethod
    def apply(self, arr: jnp.ndarray, wavelength: float):
        """Apply component effect to the input array."""
        raise NotImplementedError
