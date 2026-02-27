"""Tests for Exposure parameters in coronagraphoto.core.exposure."""

import jax.numpy as jnp

from coronagraphoto.core.exposure import Exposure


class TestExposure:
    """Tests for Exposure dataclass."""

    def test_creation(self):
        """Exposure should be creatable with required fields."""
        exposure = Exposure(
            start_time_jd=jnp.array(2460000.0),
            exposure_time_s=jnp.array(3600.0),
            central_wavelength_nm=jnp.array(600.0),
            bin_width_nm=jnp.array(100.0),
            position_angle_deg=jnp.array(0.0),
        )

        assert jnp.isclose(exposure.start_time_jd, 2460000.0)
        assert jnp.isclose(exposure.exposure_time_s, 3600.0)
        assert jnp.isclose(exposure.central_wavelength_nm, 600.0)

    def test_in_axes_default_none(self):
        """Default in_axes should have all None (broadcast)."""
        in_axes = Exposure.in_axes()

        # All fields should be None by default
        assert in_axes.start_time_jd is None
        assert in_axes.exposure_time_s is None
        assert in_axes.central_wavelength_nm is None
        assert in_axes.bin_width_nm is None
        assert in_axes.position_angle_deg is None

    def test_in_axes_with_vectorization(self):
        """in_axes should allow specifying vectorized dimensions."""
        in_axes = Exposure.in_axes(
            central_wavelength_nm=0,
            bin_width_nm=0,
        )

        # Specified fields should have axis 0
        assert in_axes.central_wavelength_nm == 0
        assert in_axes.bin_width_nm == 0

        # Other fields should be None
        assert in_axes.start_time_jd is None
        assert in_axes.exposure_time_s is None
        assert in_axes.position_angle_deg is None

    def test_vector_exposure(self):
        """Exposure should support vector values for vmap."""
        wavelengths = jnp.array([500.0, 600.0, 700.0])
        bin_widths = jnp.array([50.0, 100.0, 50.0])

        exposure = Exposure(
            start_time_jd=jnp.array(2460000.0),  # Scalar - broadcast
            exposure_time_s=jnp.array(3600.0),  # Scalar - broadcast
            central_wavelength_nm=wavelengths,  # Vector
            bin_width_nm=bin_widths,  # Vector
            position_angle_deg=jnp.array(0.0),  # Scalar - broadcast
        )

        assert exposure.central_wavelength_nm.shape == (3,)
        assert exposure.bin_width_nm.shape == (3,)
