"""Tests for OpticalPath in coronagraphoto.core.optical_path."""

import jax.numpy as jnp

from coronagraphoto.optical_elements.throughput_elements import (
    ConstantThroughputElement,
    LinearThroughputElement,
)


class TestOpticalPathAttenuation:
    """Tests for combined attenuation calculation.

    Note: Full OpticalPath testing requires a complete coronagraph setup.
    These tests verify the attenuation calculation logic in isolation.
    """

    def test_constant_elements_multiply(self):
        """Multiple constant elements should multiply."""
        elem1 = ConstantThroughputElement(0.9)
        elem2 = ConstantThroughputElement(0.8)
        elem3 = ConstantThroughputElement(0.95)

        # Manual calculation: 0.9 * 0.8 * 0.95
        expected = 0.684

        # Verify the elements work correctly
        t1 = elem1.get_throughput(550.0)
        t2 = elem2.get_throughput(550.0)
        t3 = elem3.get_throughput(550.0)

        actual = t1 * t2 * t3
        assert jnp.isclose(actual, expected, rtol=0.01)

    def test_mixed_elements(self):
        """Constant and linear elements should combine correctly."""
        const_elem = ConstantThroughputElement(0.9)

        wavelengths = jnp.array([400.0, 600.0, 800.0])
        throughputs = jnp.array([0.7, 0.9, 0.8])
        linear_elem = LinearThroughputElement(wavelengths, throughputs)

        # At 600 nm
        wavelength = 600.0
        t_const = const_elem.get_throughput(wavelength)
        t_linear = linear_elem.get_throughput(wavelength)

        combined = t_const * t_linear
        expected = 0.9 * 0.9
        assert jnp.isclose(combined, expected, rtol=0.01)

    def test_wavelength_dependence(self):
        """Combined attenuation should vary with wavelength."""
        wavelengths = jnp.array([400.0, 600.0, 800.0])
        throughputs = jnp.array([0.7, 0.9, 0.8])
        linear_elem = LinearThroughputElement(wavelengths, throughputs)

        t_blue = linear_elem.get_throughput(400.0)
        t_green = linear_elem.get_throughput(600.0)
        t_red = linear_elem.get_throughput(800.0)

        # Should have different values
        assert t_blue != t_green
        assert t_green != t_red
