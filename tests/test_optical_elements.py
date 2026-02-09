"""Tests for optical elements in coronagraphoto.optical_elements."""

import jax.numpy as jnp
import pytest

from coronagraphoto.optical_elements.optical_filters import OpticalFilter
from coronagraphoto.optical_elements.primary import PrimaryAperture
from coronagraphoto.optical_elements.throughput_elements import (
    ConstantThroughputElement,
    LinearThroughputElement,
)


class TestPrimaryAperture:
    """Tests for PrimaryAperture class."""

    def test_area_calculation(self):
        """Area should be pi * (D/2)^2 * (1 - obscuration)."""
        diameter_m = 6.0
        obscuration = 0.0

        primary = PrimaryAperture(diameter_m, obscuration)

        expected_area = jnp.pi * (diameter_m / 2) ** 2
        assert jnp.isclose(primary.area_m2, expected_area)

    def test_area_with_obscuration(self):
        """Obscuration should reduce effective area."""
        diameter_m = 6.0
        obscuration = 0.2  # 20% obscured

        primary = PrimaryAperture(diameter_m, obscuration)

        full_area = jnp.pi * (diameter_m / 2) ** 2
        expected_area = full_area * (1 - obscuration)
        assert jnp.isclose(primary.area_m2, expected_area)

    def test_apply_scales_flux(self):
        """Apply should multiply by area."""
        diameter_m = 6.0
        primary = PrimaryAperture(diameter_m)

        inc_flux = 1.0  # 1 ph/s/m^2/nm
        out_flux = primary.apply(inc_flux, wavelength=550.0)

        assert jnp.isclose(out_flux, primary.area_m2)

    def test_apply_with_array(self):
        """Apply should work with arrays."""
        primary = PrimaryAperture(6.0)
        inc_flux = jnp.ones((10, 10))

        out_flux = primary.apply(inc_flux, wavelength=550.0)

        assert out_flux.shape == (10, 10)
        assert jnp.allclose(out_flux, primary.area_m2)


class TestConstantThroughputElement:
    """Tests for ConstantThroughputElement class."""

    def test_constant_throughput(self):
        """Throughput should be constant regardless of wavelength."""
        throughput = 0.85
        element = ConstantThroughputElement(throughput)

        assert jnp.isclose(element.get_throughput(400.0), throughput)
        assert jnp.isclose(element.get_throughput(700.0), throughput)
        assert jnp.isclose(element.get_throughput(1000.0), throughput)

    def test_apply(self):
        """Apply should multiply by throughput."""
        throughput = 0.9
        element = ConstantThroughputElement(throughput)

        arr = jnp.array([100.0, 200.0, 300.0])
        result = element.apply(arr, wavelength=550.0)

        expected = arr * throughput
        assert jnp.allclose(result, expected)


class TestLinearThroughputElement:
    """Tests for LinearThroughputElement class."""

    @pytest.fixture
    def linear_element(self):
        """Create a LinearThroughputElement with known throughput curve."""
        wavelengths = jnp.array([400.0, 600.0, 800.0])
        throughputs = jnp.array([0.7, 0.9, 0.8])
        return LinearThroughputElement(wavelengths, throughputs)

    def test_interpolation_at_grid_points(self, linear_element):
        """Throughput at grid points should match input."""
        assert jnp.isclose(linear_element.get_throughput(400.0), 0.7, rtol=0.01)
        assert jnp.isclose(linear_element.get_throughput(600.0), 0.9, rtol=0.01)
        assert jnp.isclose(linear_element.get_throughput(800.0), 0.8, rtol=0.01)

    def test_interpolation_between_points(self, linear_element):
        """Throughput should interpolate between grid points."""
        # At 500 nm (midpoint of 400-600), expect ~0.8
        throughput = linear_element.get_throughput(500.0)
        assert 0.75 < throughput < 0.85

    def test_extrapolation_returns_zero(self, linear_element):
        """Throughput outside range should extrapolate to zero."""
        # Far outside range
        throughput_low = linear_element.get_throughput(200.0)
        throughput_high = linear_element.get_throughput(1200.0)

        assert jnp.isclose(throughput_low, 0.0)
        assert jnp.isclose(throughput_high, 0.0)

    def test_apply(self, linear_element):
        """Apply should multiply by interpolated throughput."""
        arr = jnp.array([100.0])
        result = linear_element.apply(arr, wavelength=600.0)
        assert jnp.isclose(result[0], 90.0, rtol=0.01)  # 100 * 0.9


class TestOpticalFilter:
    """Tests for OpticalFilter class."""

    @pytest.fixture
    def bandpass_filter(self):
        """Create a simple bandpass filter."""
        wavelengths = jnp.array([500.0, 550.0, 600.0, 650.0, 700.0])
        transmittances = jnp.array([0.0, 0.5, 1.0, 0.5, 0.0])
        return OpticalFilter(wavelengths, transmittances)

    def test_peak_transmission(self, bandpass_filter):
        """Peak transmission should be at filter center."""
        peak = bandpass_filter.get_throughput(600.0)
        assert jnp.isclose(peak, 1.0, rtol=0.01)

    def test_edge_transmission(self, bandpass_filter):
        """Edges should have lower transmission."""
        edge = bandpass_filter.get_throughput(550.0)
        assert jnp.isclose(edge, 0.5, rtol=0.01)

    def test_out_of_band_zero(self, bandpass_filter):
        """Out of band should be zero."""
        out_of_band = bandpass_filter.get_throughput(400.0)
        assert jnp.isclose(out_of_band, 0.0)

    def test_apply(self, bandpass_filter):
        """Apply should multiply by transmittance."""
        arr = jnp.array([1000.0])
        result = bandpass_filter.apply(arr, wavelength_nm=600.0)
        assert jnp.isclose(result[0], 1000.0, rtol=0.01)  # 1000 * 1.0
