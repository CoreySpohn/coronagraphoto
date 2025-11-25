"""Defines the exposure parameters for a coronagraphoto simulation."""

from dataclasses import fields

import equinox as eqx
import jax.numpy as jnp


class Exposure(eqx.Module):
    """The physical parameters defining a single detector integration.

    All fields can be scalars (for a single event) or vectors (for a sequence),
    depending on how the factories are composed.
    """

    start_time_jd: jnp.ndarray  # Julian Date
    exposure_time_s: jnp.ndarray  # Seconds
    central_wavelength_nm: jnp.ndarray  # Nanometers
    bin_width_nm: jnp.ndarray  # Nanometers
    position_angle_deg: jnp.ndarray  # Degrees

    @classmethod
    def in_axes(cls, **vectorized_axes):
        """Helper to generate the in_axes structure for JAX vmap over an Exposure.

        Usage:
            # Vectorize over wavelength (axis 0), keep time constant
            in_axes = Exposure.in_axes(central_wavelength_nm=0, bin_width_nm=0)
        """
        # Default all fields to None (Broadcast/Constant)
        spec_dict = {f.name: None for f in fields(cls)}

        # Update specific fields to be vectorized
        spec_dict.update(vectorized_axes)

        return cls(**spec_dict)
