"""Astrophysical source objects — aliases for exoverses.jax types.

These names are kept for backwards compatibility. New code should import
directly from ``exoverses.jax``.
"""

from exoverses.jax import Disk as DiskSource  # noqa: F401
from exoverses.jax import Planet as PlanetSources  # noqa: F401
from exoverses.jax import Star as StarSource  # noqa: F401

# Re-export the abstract base class as well (kept here for type annotations)
# The simulation code doesn't import AbstractSource from here, but it's part
# of the public API. We keep a minimal version for type-checking purposes.
import abc
import equinox as eqx
import jax.numpy as jnp


class AbstractSource(eqx.Module):
    """Abstract base class defining the interface for all astrophysical sources.

    All sources accept scalar wavelength and time inputs. Use jax.vmap
    for vectorized evaluation over multiple wavelengths/times.
    """

    @abc.abstractmethod
    def spec_flux_density(self, wavelength: float, time: float) -> float | jnp.ndarray:
        """Return spectral flux density in ph/s/m^2/nm."""
        raise NotImplementedError

